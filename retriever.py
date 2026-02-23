"""
retriever.py
============
Phase 3: Hybrid Retrieval สำหรับ Movex Product Chatbot

รับ TransformResult จาก query_transform.py แล้ว:
  1. Search text collection ด้วยทุก query (Jina v3)
  2. Search image collection ด้วย ViT LoRA + SigLIP (ถ้ามีรูป)
  3. ใช้ ViT LoRA เป็น specialist gating — narrow pool ก่อน search
  4. รวม score ด้วย Reciprocal Rank Fusion (RRF)
  5. คืน ranked product list พร้อม metadata ครบ

ใช้งาน:
  from retriever import Retriever
  retriever = Retriever(cfg)
  results = retriever.search(transform_result, query_image=pil_img)
"""

import os
import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Models (โหลดจากภายนอก เพื่อไม่ต้องโหลดซ้ำ)
from transformers import (
    AutoTokenizer, AutoModel,
    ViTImageProcessor, ViTForImageClassification,
)
from transformers import SiglipModel, SiglipProcessor
from peft import PeftModel

# Import จาก phase 2
try:
    from query_transform import TransformResult
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class TransformResult:
        original_query: str
        all_queries: list = field(default_factory=list)
        query_type: str = "general"   # รับ query_type จาก QueryTransformer
        def __post_init__(self):
            if not self.all_queries:
                self.all_queries = [self.original_query]

# ===============================================================
# DATA CLASSES
# ===============================================================

@dataclass
class RetrievedProduct:
    """ผลลัพธ์จาก retrieval ต่อ 1 product"""
    product_id: str
    rrf_score: float                          # score หลังทำ RRF fusion
    matched_chunks: list[dict] = field(default_factory=list)  # text chunks ที่ match
    matched_images: list[dict] = field(default_factory=list)  # รูปที่ match
    series: str = ""
    product_type: str = ""
    material: str = ""
    # scores แยกตามแหล่ง (สำหรับ debug)
    text_rank: Optional[int] = None
    image_vit_rank: Optional[int] = None
    image_siglip_rank: Optional[int] = None
    keyword_boost: float = 0.0


@dataclass
class RetrievalResult:
    """ผลลัพธ์ทั้งหมดจาก retrieval pipeline"""
    products: list[RetrievedProduct]
    query_used: str                   # query ที่ใช้ search จริง
    total_candidates: int = 0
    image_search_used: bool = False
    vit_series_filter: Optional[str] = None  # series ที่ ViT detect ได้

    def top(self, k: int = 5) -> list[RetrievedProduct]:
        return self.products[:k]


# ===============================================================
# RETRIEVER CONFIG
# ===============================================================

@dataclass
class RetrieverConfig:
    qdrant_path: str = "./qdrant_db"
    text_collection: str = "movex_text_chunks"
    image_collection: str = "movex_images"

    # Search params
    text_candidates: int = 20         # ดึงมาก่อน rerank
    image_candidates: int = 20
    final_top_k: int = 5

    # RRF constant (k=60 มาตรฐาน)
    rrf_k: int = 60

    # Weights ต่อ search type ใน RRF
    weight_text: float = 1.0
    weight_vit: float = 1.2           # ให้ specialist weight สูงกว่า
    weight_siglip: float = 0.8
    weight_keyword: float = 2.0       # exact match สำคัญมาก

    # ViT gating threshold
    vit_confidence_threshold: float = 0.50  # ลดจาก 0.70 เพื่อให้ detect series ได้ง่ายขึ้น

    # Image relevance gate (ถ้าใกล้เกินไปแปลว่า irrelevant)
    image_distance_threshold: float = 0.64  # Cosine score >= 0.64 = L2 dist <= 0.85 (เทียบเท่า ChromaDB เดิม)

    # Text dim
    jina_dim: int = 1024
    vit_dim: int = 768
    siglip_dim: int = 768

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Series ที่รู้จัก
    known_series: set = field(
        default_factory=lambda: {"820", "821", "880", "882", "83", "103"}
    )

    # Spec field mapping: ชื่อ abstract → payload key ใน Qdrant
    # ใช้สำหรับ filter_by_spec และ LLM tool calls
    # ครอบคลุมทั้ง chain และ sprocket
    spec_field_map: dict = field(default_factory=lambda: {
        # Chain
        "width":  "plate_width_mm",
        "load":   "max_load_n",
        "weight": "weight_kg_m",
        "radius": "min_curve_radius_mm",
        "pitch":  "pitch_mm",
        # Sprocket
        "teeth":  "z_teeth",
        "bore":   "bore_mm",
        "pd":     "pd_mm",
        "od":     "od_mm",
        "s":      "s_mm",
    })


# ===============================================================
# RETRIEVER
# ===============================================================

class Retriever:
    """
    Hybrid retriever รวม text + image search ด้วย RRF

    Parameters
    ----------
    cfg : RetrieverConfig
    jina_tokenizer, jina_model : โหลดมาจาก build_database.py / main app
    vit_processor, vit_model   : ViT LoRA specialist
    siglip_processor, siglip_model : SigLIP generalist
    product_db : dict[product_id → product dict] จาก JSON
    """

    PRODUCT_ID_PATTERN = re.compile(
        r'\b(LF|SS|POM|PP|820|821|880|882|83|103|K\d{3})\b', re.IGNORECASE
    )

    def __init__(
        self,
        cfg: RetrieverConfig,
        jina_tokenizer,
        jina_model,
        vit_processor,
        vit_model,
        siglip_processor,
        siglip_model,
        product_db: dict,
    ):
        self.cfg = cfg
        self.client = QdrantClient(path=cfg.qdrant_path)

        self.jina_tokenizer = jina_tokenizer
        self.jina_model = jina_model
        self.vit_processor = vit_processor
        self.vit_model = vit_model
        self.siglip_processor = siglip_processor
        self.siglip_model = siglip_model
        self.product_db = product_db

    # -----------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------

    def search(
        self,
        transform_result: TransformResult,
        query_image: Optional[Image.Image] = None,
    ) -> RetrievalResult:
        """
        รับ TransformResult + optional image
        คืน RetrievalResult เรียงตาม RRF score
        """
        result = RetrievalResult(
            products=[],
            query_used=transform_result.original_query,
        )

        # scores[product_id] = {"text_ranks": [], "vit_rank": None, "siglip_rank": None, "keyword": 0}
        score_map: dict[str, dict] = {}

        # ── 3. Image Search ก่อน เพื่อ detect vit_dominant ────
        vit_series = None
        vit_dominant = False
        query_type = getattr(transform_result, "query_type", "general")
        if query_image is not None:
            result.image_search_used = True
            vit_series, vit_dominant = self._image_search(query_image, score_map)
            result.vit_series_filter = vit_series
        elif query_type == "visual":
            result.image_search_used = True
            self._text_to_image_search(transform_result.original_query, score_map)

        # ── 1. Keyword Boost ──────────────────────────────────
        self._keyword_boost(transform_result.original_query, score_map)

        # ── 2. Text Search ────────────────────────────────────
        # ถ้า ViT dominant → skip text search ป้องกัน text override image
        if not vit_dominant:
            self._multi_query_text_search(transform_result.all_queries, score_map)

        # ── 4. RRF Fusion ─────────────────────────────────────
        ranked = self._rrf_fusion(score_map)

        # ── 5. Build RetrievedProduct list ────────────────────
        products = []
        for pid, rrf_score in ranked[:self.cfg.final_top_k * 2]:
            p_data = self.product_db.get(pid)
            if not p_data:
                continue

            rp = RetrievedProduct(
                product_id=pid,
                rrf_score=rrf_score,
                series=p_data.get("series", ""),
                product_type=p_data.get("product_type", ""),
                material=p_data.get("Material", ""),
                keyword_boost=score_map.get(pid, {}).get("keyword", 0.0),
            )

            # ดึง chunk texts ที่ match มาแนบ
            rp.matched_chunks = self._fetch_chunk_texts(pid)

            products.append(rp)

        result.products = products[:self.cfg.final_top_k]
        result.total_candidates = len(ranked)
        return result

    # -----------------------------------------------------------
    # KEYWORD BOOST
    # -----------------------------------------------------------

    def _keyword_boost(self, query: str, score_map: dict):
        """
        Exact / fuzzy match กับ product_id และ series
        ให้ boost score สูงเพื่อให้ product ที่ชื่อตรงติด top เสมอ
        """
        import difflib

        query_clean = re.sub(r'[^a-zA-Z0-9]', '', query).lower()
        query_lower = query.lower()

        # ตรวจ keyword ก่อน early return เพราะ query ภาษาไทยทำให้ query_clean ว่าง
        sideflex_kw = any(w in query_lower for w in [
            "เลี้ยว", "โค้ง", "sideflex", "sideflexing", "curve", "curving"
        ])
        straight_kw = any(w in query_lower for w in ["ตรง", "straight", "วิ่งตรง"])

        # ถ้า query เป็นภาษาไทยทั้งหมด query_clean จะว่าง
        # '' in any_string = True → ทุก product ได้ boost 10.0 → ผิดทั้งหมด
        # แต่ถ้ามี sideflex/straight keyword ยังต้องทำงานต่อ
        if not query_clean and not sideflex_kw and not straight_kw:
            return

        query_nums   = re.findall(r'\d+', query_clean)
        query_series = [n for n in query_nums if n in self.cfg.known_series]

        for pid, p in self.product_db.items():
            pid_clean   = re.sub(r'[^a-zA-Z0-9]', '', pid).lower()
            ref_clean   = re.sub(r'[^a-zA-Z0-9]', '', str(p.get("Ref", ""))).lower()
            artnr_clean = re.sub(r'[^a-zA-Z0-9]', '', str(p.get("Art_Nr", ""))).lower()
            short_pid   = pid_clean.replace("movexchain", "").replace("movexsprocket", "")
            series_str  = re.sub(r'[^a-zA-Z0-9]', '', str(p.get("series", ""))).lower()
            ptype       = str(p.get("product_type", "")).lower()

            # Series conflict check
            if query_series:
                prod_series_nums = re.findall(r'\d+', series_str)
                if not any(qs in prod_series_nums for qs in query_series):
                    continue

            boost = 0.0

            # Product type keyword boost
            if sideflex_kw and "sideflexing chain" in ptype:
                boost = max(boost, 8.0)
            if straight_kw and "straight running" in ptype:
                boost = max(boost, 8.0)

            # Exact/fuzzy match — ทำเฉพาะเมื่อ query_clean ไม่ว่าง
            # ถ้าว่าง '' in any_string = True → ทุกตัวได้ boost
            if query_clean:
                if query_clean in (ref_clean, pid_clean, short_pid, artnr_clean):
                    boost = 10.0
                elif boost == 0.0:
                    sim = max(
                        difflib.SequenceMatcher(None, query_clean, ref_clean).ratio(),
                        difflib.SequenceMatcher(None, query_clean, short_pid).ratio(),
                        difflib.SequenceMatcher(None, query_clean, artnr_clean).ratio(),
                    )
                    if sim > 0.85:
                        boost = 6.0
                    elif sim > 0.70:
                        boost = 3.0
                    elif (query_clean in pid_clean or query_clean in ref_clean
                          or query_clean in artnr_clean):
                        boost = 2.0

            if boost > 0:
                if pid not in score_map:
                    score_map[pid] = {"text_ranks": [], "vit_rank": None,
                                      "siglip_rank": None, "keyword": 0.0}
                score_map[pid]["keyword"] = max(score_map[pid]["keyword"], boost)

    # -----------------------------------------------------------
    # TEXT SEARCH
    # -----------------------------------------------------------

    def _multi_query_text_search(self, queries: list[str], score_map: dict):
        """
        Search ด้วยทุก query จาก TransformResult.all_queries
        เก็บ rank ของแต่ละ product ต่อแต่ละ query แยกกัน
        RRF จะรวมทีหลัง
        """
        for query in queries:
            vec = self._embed_text(query)
            if vec is None:
                continue

            result = self.client.query_points(
                collection_name=self.cfg.text_collection,
                query=vec,
                using="jina",
                limit=self.cfg.text_candidates,
                with_payload=True,
            )
            hits = result.points

            for rank, hit in enumerate(hits):
                pid = hit.payload.get("product_id", "")
                if not pid:
                    continue
                if pid not in score_map:
                    score_map[pid] = {"text_ranks": [], "vit_rank": None,
                                      "siglip_rank": None, "keyword": 0.0}
                score_map[pid]["text_ranks"].append(rank)

    # -----------------------------------------------------------
    # IMAGE SEARCH
    # -----------------------------------------------------------

    def _image_search(
        self,
        image: Image.Image,
        score_map: dict,
    ) -> Optional[str]:
        """
        Dual image search:
        1. ViT LoRA → specialist, detect series → ใช้เป็น filter pool
        2. SigLIP   → generalist, semantic match

        คืน series ที่ ViT detect ได้ (หรือ None)
        """
        image = image.convert("RGB")

        # ── ViT LoRA ──────────────────────────────────────────
        vit_vec = self._embed_vit(image)
        vit_series = None

        if vit_vec:
            _vit_res = self.client.query_points(
                collection_name=self.cfg.image_collection,
                query=vit_vec,
                using="vit",
                limit=self.cfg.image_candidates,
                with_payload=True,
            )
            vit_hits = _vit_res.points

            # กรอง irrelevant images
            # Cosine similarity: 1.0 = identical, 0.0 = unrelated
            # กรองเฉพาะ hits ที่ score ต่ำเกินไป (irrelevant)
            valid_vit = [h for h in vit_hits if h.score >= self.cfg.image_distance_threshold]

            # ถ้าไม่มี hit ที่ดีพอ ให้ใช้ top-3 แทน (ดีกว่าคืน None)
            if not valid_vit:
                valid_vit = vit_hits[:3] if vit_hits else []

            if not valid_vit:
                return None, False  # ไม่มี hits เลย

            # ตรวจ series จาก top hit ถ้า confidence สูงพอ
            top_hit = valid_vit[0]
            if top_hit.score >= self.cfg.vit_confidence_threshold:
                vit_series = top_hit.payload.get("series", "")
                # แปลง "820 Series" → "820"
                series_num = re.search(r'\d+', vit_series or "")
                vit_series = series_num.group() if series_num else None

            for rank, hit in enumerate(valid_vit):
                pid = hit.payload.get("product_id", "")
                if not pid:
                    continue
                if pid not in score_map:
                    score_map[pid] = {"text_ranks": [], "vit_rank": None,
                                      "siglip_rank": None, "keyword": 0.0}
                if score_map[pid]["vit_rank"] is None:
                    score_map[pid]["vit_rank"] = rank

            # ── ViT Dominant Mode ──────────────────────────────
            # ถ้า ViT top score >= 0.85 แปลว่า ViT มั่นใจมาก
            # skip SigLIP และ text search ป้องกัน text override image
            if top_hit.score >= 0.85:
                return vit_series, True  # (series, vit_dominant=True)

        # ── SigLIP (รันเฉพาะเมื่อ ViT ไม่มั่นใจ) ────────────
        siglip_vec = self._embed_siglip(image)

        if siglip_vec:
            # ถ้า ViT detect series ได้ → filter เฉพาะ series นั้น
            qdrant_filter = None
            if vit_series:
                qdrant_filter = Filter(
                    must=[FieldCondition(
                        key="series",
                        match=MatchValue(value=f"{vit_series} Series")
                    )]
                )

            _siglip_res = self.client.query_points(
                collection_name=self.cfg.image_collection,
                query=siglip_vec,
                using="siglip",
                limit=self.cfg.image_candidates,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            siglip_hits = _siglip_res.points

            for rank, hit in enumerate(siglip_hits):
                pid = hit.payload.get("product_id", "")
                if not pid:
                    continue
                if pid not in score_map:
                    score_map[pid] = {"text_ranks": [], "vit_rank": None,
                                      "siglip_rank": None, "keyword": 0.0}
                if score_map[pid]["siglip_rank"] is None:
                    score_map[pid]["siglip_rank"] = rank

        return vit_series, False  # vit_dominant=False

    # -----------------------------------------------------------
    # RRF FUSION
    # -----------------------------------------------------------

    def _rrf_fusion(self, score_map: dict) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion รวม signals ทุกแหล่ง

        RRF score = Σ weight_i / (k + rank_i)

        Text: รวม rank จากทุก query (multi-query) แล้วเฉลี่ย
        ViT / SigLIP: rank จาก image search
        Keyword: เพิ่มตรงๆ เป็น bonus
        """
        k = self.cfg.rrf_k
        final_scores = {}

        for pid, scores in score_map.items():
            rrf = 0.0

            # Text: เฉลี่ย RRF จากทุก query
            text_ranks = scores.get("text_ranks", [])
            if text_ranks:
                text_rrf = sum(
                    self.cfg.weight_text / (k + r) for r in text_ranks
                ) / len(text_ranks)
                rrf += text_rrf

            # ViT
            vit_rank = scores.get("vit_rank")
            if vit_rank is not None:
                rrf += self.cfg.weight_vit / (k + vit_rank)

            # SigLIP
            siglip_rank = scores.get("siglip_rank")
            if siglip_rank is not None:
                rrf += self.cfg.weight_siglip / (k + siglip_rank)

            # Keyword boost (เพิ่มตรงๆ เพราะ exact match ไม่มี rank)
            keyword = scores.get("keyword", 0.0)
            if keyword > 0:
                rrf += self.cfg.weight_keyword * keyword / 10.0  # normalize

            final_scores[pid] = rrf

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # -----------------------------------------------------------
    # HELPERS: EMBEDDING
    # -----------------------------------------------------------

    @torch.no_grad()
    def _embed_text(self, text: str) -> Optional[list[float]]:
        try:
            enc = self.jina_tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512
            ).to(self.cfg.device)
            out = self.jina_model(**enc, return_dict=True)
            # Mean pool
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(1) / mask.sum(1)
            pooled = F.normalize(pooled, dim=-1)
            return pooled.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ Text embed error: {e}")
            return None

    @torch.no_grad()
    def _embed_vit(self, image: Image.Image) -> Optional[list[float]]:
        try:
            inputs = self.vit_processor(
                images=image, return_tensors="pt"
            ).to(self.cfg.device)
            out = self.vit_model.base_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True, return_dict=True,
            )
            cls = out.hidden_states[-1][:, 0, :]
            cls = F.normalize(cls, p=2, dim=1)
            return cls.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ ViT embed error: {e}")
            return None

    @torch.no_grad()
    def _embed_siglip(self, image: Image.Image) -> Optional[list[float]]:
        try:
            inputs = self.siglip_processor(
                images=image, return_tensors="pt"
            ).to(self.cfg.device)
            feat = self.siglip_model.get_image_features(**inputs)
            feat = F.normalize(feat, dim=-1)
            return feat.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ SigLIP embed error: {e}")
            return None

    @torch.no_grad()
    def _embed_siglip_text(self, text: str) -> Optional[list[float]]:
        """
        SigLIP text encoder — embed ข้อความลงใน image embedding space
        ทำให้ค้นหา image vectors ด้วยคำอธิบายรูปร่างได้
        เช่น "สายพานสีน้ำตาล บานพับเดียว" → หา image ที่คล้ายกัน
        """
        try:
            inputs = self.siglip_processor(
                text=[text], return_tensors="pt", padding=True,
                truncation=True, max_length=64
            ).to(self.cfg.device)
            feat = self.siglip_model.get_text_features(**inputs)
            feat = F.normalize(feat, dim=-1)
            return feat.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ SigLIP text embed error: {e}")
            return None

    def _text_to_image_search(self, text: str, score_map: dict):
        """
        ลูกค้าอธิบายรูปร่างสินค้าด้วยคำพูด (ไม่มีรูป)
        → SigLIP embed ข้อความ → ค้นหาใน siglip image vectors
        ใช้เฉพาะเมื่อ query_type == "visual"
        """
        siglip_vec = self._embed_siglip_text(text)
        if not siglip_vec:
            return

        result = self.client.query_points(
            collection_name=self.cfg.image_collection,
            query=siglip_vec,
            using="siglip",
            limit=self.cfg.image_candidates,
            with_payload=True,
        )

        for rank, hit in enumerate(result.points):
            pid = hit.payload.get("product_id", "")
            if not pid:
                continue
            # text-image gap ทำให้ score ต่ำกว่า image-image
            # ใช้ threshold ต่ำกว่า image search
            if hit.score < 0.20:
                continue
            if pid not in score_map:
                score_map[pid] = {"text_ranks": [], "vit_rank": None,
                                  "siglip_rank": None, "keyword": 0.0}
            if score_map[pid]["siglip_rank"] is None:
                score_map[pid]["siglip_rank"] = rank

    # -----------------------------------------------------------
    # HELPERS: FETCH CHUNK TEXTS
    # -----------------------------------------------------------

    def _fetch_chunk_texts(self, product_id: str) -> list[dict]:
        """
        ดึง chunk texts ทั้งหมดของ product จาก Qdrant
        สำหรับใส่ใน LLM context
        """
        try:
            hits, _ = self.client.scroll(
                collection_name=self.cfg.text_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="product_id",
                        match=MatchValue(value=product_id)
                    )]
                ),
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            return [
                {"type": h.payload.get("chunk_type"), "text": h.payload.get("text")}
                for h in hits
            ]
        except Exception:
            return []