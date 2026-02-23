"""
retriever_v3.py
===============
Phase 3 (v3): Hybrid Retrieval + GraphRAG สำหรับ Movex Product Chatbot

เปลี่ยนแปลงหลักจาก retriever.py (v2):
  ✦ ลบ dependency ต่อ QueryTransformer / TransformResult ออกทั้งหมด
    → รับ query: str ธรรมดาแทน (ไม่ expand / HyDE)
  ✦ เพิ่ม GraphRAG step หลัง ViT ระบุ series / product ได้
    → expand_products() ดึง same_series, compatible sprockets, material details
    → เพิ่ม graph_candidates เข้า score_map ก่อน RRF

Pipeline v3:
  query (str) + optional image
        ↓
  1. Keyword Boost          — exact/fuzzy match product_id, ref, art_nr
        ↓
  2. Image Search           — ViT LoRA (specialist) + SigLIP (generalist)
        ↓
  3. GraphRAG Expansion     — expand via KnowledgeGraph หลัง ViT detect
        ↓
  4. Text Search            — Jina v3 single-query (ไม่ expand)
        ↓
  5. RRF Fusion             — รวม scores ทุก signal
        ↓
  6. Build RetrievedProduct list พร้อม GraphContext

ใช้งาน:
  from retriever_v3 import RetrieverV3, RetrieverV3Config
  retriever = RetrieverV3(cfg, jina_tok, jina_model, vit_proc, vit_model,
                          siglip_proc, siglip_model, product_db, kg)
  result = retriever.search("สายพาน 880 เลี้ยวได้", query_image=img)
"""

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Models
from transformers import (
    AutoTokenizer, AutoModel,
    ViTImageProcessor, ViTForImageClassification,
    SiglipModel, SiglipProcessor,
)
from peft import PeftModel

# GraphRAG
from knowledge_graph import KnowledgeGraph, GraphContext

# Re-use dataclasses จาก retriever.py เดิม
from retriever import RetrievedProduct, RetrievalResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXTENDED RESULT — เพิ่ม GraphContext
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RetrievalResultV3(RetrievalResult):
    """
    เหมือน RetrievalResult เดิม + graph_context สำหรับ LLM
    """
    graph_context: Optional[GraphContext] = None

    def top(self, k: int = 5) -> list[RetrievedProduct]:
        return self.products[:k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RetrieverV3Config:
    qdrant_path: str = "./qdrant_db"
    text_collection: str = "movex_text_chunks"
    image_collection: str = "movex_images"

    # Search params
    text_candidates: int = 20
    image_candidates: int = 20
    final_top_k: int = 10     # คืนมากขึ้นให้ reranker เลือก

    # RRF constant
    rrf_k: int = 60

    # Weights ต่อ signal ใน RRF
    weight_text: float    = 1.0
    weight_vit: float     = 1.2   # specialist สำคัญกว่า
    weight_siglip: float  = 0.8
    weight_keyword: float = 2.0   # exact match สำคัญมาก
    weight_graph: float   = 0.6   # graph candidates ได้ boost เพิ่ม

    # ViT thresholds
    vit_confidence_threshold: float = 0.50
    image_distance_threshold: float = 0.64

    # Embedding dims
    jina_dim: int    = 1024
    vit_dim: int     = 768
    siglip_dim: int  = 768

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Series ที่รู้จัก
    known_series: set = field(
        default_factory=lambda: {"820", "821", "880", "882", "83", "103"}
    )

    # GraphRAG options
    graph_expand_series: bool      = True   # expand same-series products
    graph_expand_compatible: bool  = True   # expand compatible sprockets/chains
    graph_expand_material: bool    = False  # expand same-material (กว้างเกินปิดไว้)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RETRIEVER V3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RetrieverV3:
    """
    Hybrid Retriever v3 — Text + Image + GraphRAG

    Parameters
    ----------
    cfg : RetrieverV3Config
    jina_tokenizer, jina_model : Jina v3 text encoder
    vit_processor, vit_model   : ViT LoRA specialist
    siglip_processor, siglip_model : SigLIP generalist
    product_db : dict[product_id → product dict]
    kg : KnowledgeGraph — สร้างไว้แล้วตอน startup
    """

    PRODUCT_ID_RE = re.compile(
        r'\b(LF|LFN|SS|POM|PP|820|821|880|882|83|103|K\d{3,4})\b',
        re.IGNORECASE
    )

    def __init__(
        self,
        cfg: RetrieverV3Config,
        jina_tokenizer,
        jina_model,
        vit_processor,
        vit_model,
        siglip_processor,
        siglip_model,
        product_db: dict,
        kg: KnowledgeGraph,
    ):
        self.cfg             = cfg
        self.client          = QdrantClient(path=cfg.qdrant_path)
        self.jina_tokenizer  = jina_tokenizer
        self.jina_model      = jina_model
        self.vit_processor   = vit_processor
        self.vit_model       = vit_model
        self.siglip_processor = siglip_processor
        self.siglip_model    = siglip_model
        self.product_db      = product_db
        self.kg              = kg

    # ──────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        query_type: str = "general",    # "general" | "visual" | "image"
    ) -> RetrievalResultV3:
        """
        รับ query string + optional image แล้วคืน RetrievalResultV3

        Parameters
        ----------
        query : str
            User query ดิบๆ ไม่ต้องผ่าน QueryTransformer แล้ว
        query_image : PIL.Image | None
        query_type : str
            "image"   = มีรูปจริงแนบมา
            "visual"  = user อธิบายรูปร่างด้วยคำพูด
            "general" = ข้อความปกติ

        Returns
        -------
        RetrievalResultV3
        """
        result = RetrievalResultV3(
            products=[],
            query_used=query,
        )

        # score_map[product_id] = {
        #   "text_ranks": [], "vit_rank": None,
        #   "siglip_rank": None, "keyword": 0.0, "graph": 0.0
        # }
        score_map: dict[str, dict] = {}

        # ── 1. Image Search + GraphRAG ────────────────────
        vit_series    = None
        vit_dominant  = False
        graph_context = None

        if query_image is not None:
            result.image_search_used = True
            vit_series, vit_dominant = self._image_search(query_image, score_map)
            result.vit_series_filter = vit_series

            # ── GraphRAG: expand หลัง ViT detect series ──
            if vit_series or vit_dominant:
                graph_context = self._graph_expand_from_vit(
                    vit_series=vit_series,
                    score_map=score_map,
                )

        elif query_type == "visual":
            # user อธิบายรูปด้วยคำพูด → SigLIP text→image search
            result.image_search_used = True
            self._text_to_image_search(query, score_map)

        # ── 2. Keyword Boost ──────────────────────────────
        # ทำหลัง image search เพื่อรู้ vit_dominant ก่อน
        # ถ้า vit_dominant=True → suppress sideflex/straight type boost
        # ป้องกัน "ส่งรูป 820 + ถาม เลี้ยวได้ไหม" → boost LF880/882 ผิดพลาด
        self._keyword_boost(query, score_map, vit_dominant=vit_dominant)

        # ── GraphRAG: expand จาก keyword boost (ถ้ายังไม่มี context) ──
        if graph_context is None:
            graph_context = self._graph_expand_from_keywords(
                query=query,
                score_map=score_map,
            )

        result.graph_context = graph_context

        # ── 3. Text Search (single query, ไม่ expand) ────
        # ViT dominant → skip text เพื่อไม่ให้ override image signal
        if not vit_dominant:
            self._text_search(query, score_map)

        # ── 4. RRF Fusion ─────────────────────────────────
        ranked = self._rrf_fusion(score_map)

        # ── 5. Build RetrievedProduct list ───────────────
        products: list[RetrievedProduct] = []
        target_k = self.cfg.final_top_k * 2   # ดึงเผื่อให้ reranker ตัด

        for pid, rrf_score in ranked[:target_k]:
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
            rp.matched_chunks = self._fetch_chunk_texts(pid)
            products.append(rp)

        result.products        = products[:self.cfg.final_top_k]
        result.total_candidates = len(ranked)
        return result

    # ──────────────────────────────────────────────────────
    # GRAPH EXPANSION
    # ──────────────────────────────────────────────────────

    def _graph_expand_from_vit(
        self,
        vit_series: Optional[str],
        score_map: dict,
    ) -> GraphContext:
        """
        ExpandGraphRAG หลัง ViT ระบุ series ได้

        Logic:
          1. ดึง seed products จาก score_map ที่มี vit_rank (ViT hit top products)
          2. expand ด้วย KnowledgeGraph
          3. เพิ่ม graph candidates เข้า score_map ด้วย weight_graph bonus
        """
        # seed = products ที่ ViT hit แล้ว
        seed_ids = [
            pid for pid, s in score_map.items()
            if s.get("vit_rank") is not None
        ]

        # fallback: ใช้ series ที่ ViT detect ถ้า seed ว่าง
        if not seed_ids and vit_series:
            seed_ids = self.kg.get_products_by_series(vit_series)[:5]

        if not seed_ids:
            return self.kg.expand_products([])

        ctx = self.kg.expand_products(
            seed_product_ids=seed_ids,
            include_same_series=self.cfg.graph_expand_series,
            include_compatible=self.cfg.graph_expand_compatible,
            include_same_material=self.cfg.graph_expand_material,
        )

        # เพิ่ม graph candidates เข้า score_map
        graph_rank = 0
        for pid in ctx.all_candidate_ids:
            if pid not in score_map:
                score_map[pid] = {
                    "text_ranks": [], "vit_rank": None,
                    "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                }
            # graph_rank ใช้เป็น rank สำหรับคำนวณ RRF (ยิ่งต่ำยิ่งดี)
            if score_map[pid].get("graph", 0.0) == 0.0:
                score_map[pid]["graph"] = graph_rank
                graph_rank += 1

        return ctx

    def _graph_expand_from_keywords(
        self,
        query: str,
        score_map: dict,
    ) -> GraphContext:
        """
        GraphRAG expand จาก keyword boost hits
        (ใช้เมื่อไม่มีรูป หรือ ViT ไม่ detect ได้)

        seed = products ที่ได้ keyword boost > 0 เรียงตาม boost สูงสุด
        """
        # เรียง products ที่มี keyword boost
        keyword_hits = sorted(
            [(pid, s["keyword"]) for pid, s in score_map.items() if s.get("keyword", 0) > 0],
            key=lambda x: x[1],
            reverse=True,
        )

        seed_ids = [pid for pid, _ in keyword_hits[:5]]

        if not seed_ids:
            return self.kg.expand_products([])

        ctx = self.kg.expand_products(
            seed_product_ids=seed_ids,
            include_same_series=self.cfg.graph_expand_series,
            include_compatible=self.cfg.graph_expand_compatible,
            include_same_material=self.cfg.graph_expand_material,
        )

        # เพิ่ม compatible และ same_series เข้า score_map ด้วย graph boost
        # (เฉพาะที่ยังไม่อยู่ใน score_map)
        graph_rank = 0
        for pid in ctx.compatible_products + ctx.same_series:
            if pid not in score_map:
                score_map[pid] = {
                    "text_ranks": [], "vit_rank": None,
                    "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                }
            if score_map[pid].get("graph", 0.0) == 0.0:
                score_map[pid]["graph"] = graph_rank
                graph_rank += 1

        return ctx

    # ──────────────────────────────────────────────────────
    # KEYWORD BOOST
    # ──────────────────────────────────────────────────────

    def _keyword_boost(self, query: str, score_map: dict, vit_dominant: bool = False):
        """
        Exact / fuzzy match กับ product_id, Ref, Art_Nr
        พร้อม product_type keyword สำหรับ sideflexing/straight

        vit_dominant=True → suppress sideflex/straight type boost
        เพราะ ViT บอก series ชัดแล้ว ไม่ควรให้ keyword type override
        เช่น ส่งรูป 820 + ถาม "เลี้ยวได้ไหม" → ไม่ boost LF880/882
        """
        import difflib

        query_clean = re.sub(r'[^a-zA-Z0-9]', '', query).lower()
        query_lower = query.lower()

        # Product type keywords
        sideflex_kw = any(w in query_lower for w in [
            "เลี้ยว", "โค้ง", "sideflex", "sideflexing", "curve", "curving"
        ])
        straight_kw = any(w in query_lower for w in ["ตรง", "straight", "วิ่งตรง"])

        # ภาษาไทยล้วน และไม่มี type keyword → ข้าม
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

            # ตรวจว่า query ระบุ series แต่ product อยู่ series อื่น → ข้าม
            if query_series:
                prod_series_nums = re.findall(r'\d+', series_str)
                if not any(qs in prod_series_nums for qs in query_series):
                    continue

            boost = 0.0

            # Product type keyword boost
            # vit_dominant=True → skip เพราะ ViT บอก series แล้ว
            if not vit_dominant:
                if sideflex_kw and "sideflexing chain" in ptype:
                    boost = max(boost, 8.0)
                if straight_kw and "straight running" in ptype:
                    boost = max(boost, 8.0)

            # Exact / fuzzy match (เฉพาะเมื่อ query_clean ไม่ว่าง — ทำเสมอ ไม่ขึ้นกับ vit_dominant)
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
                    score_map[pid] = {
                        "text_ranks": [], "vit_rank": None,
                        "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                    }
                score_map[pid]["keyword"] = max(score_map[pid]["keyword"], boost)

    # ──────────────────────────────────────────────────────
    # TEXT SEARCH
    # ──────────────────────────────────────────────────────

    def _text_search(self, query: str, score_map: dict):
        """
        Single-query text search ด้วย Jina v3
        (ไม่ expand query แล้ว — เอา QueryTransformer ออก)
        """
        vec = self._embed_text(query)
        if vec is None:
            return

        result = self.client.query_points(
            collection_name=self.cfg.text_collection,
            query=vec,
            using="jina",
            limit=self.cfg.text_candidates,
            with_payload=True,
        )

        for rank, hit in enumerate(result.points):
            pid = hit.payload.get("product_id", "")
            if not pid:
                continue
            if pid not in score_map:
                score_map[pid] = {
                    "text_ranks": [], "vit_rank": None,
                    "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                }
            score_map[pid]["text_ranks"].append(rank)

    # ──────────────────────────────────────────────────────
    # IMAGE SEARCH
    # ──────────────────────────────────────────────────────

    def _image_search(
        self,
        image: Image.Image,
        score_map: dict,
    ) -> tuple[Optional[str], bool]:
        """
        Dual image search:
          1. ViT LoRA → detect series → filter pool
          2. SigLIP   → semantic match (รันเมื่อ ViT ไม่ dominant)

        Returns
        -------
        (vit_series, vit_dominant)
        """
        image = image.convert("RGB")

        vit_vec    = self._embed_vit(image)
        vit_series = None

        if vit_vec:
            _vit_res  = self.client.query_points(
                collection_name=self.cfg.image_collection,
                query=vit_vec,
                using="vit",
                limit=self.cfg.image_candidates,
                with_payload=True,
            )
            vit_hits = _vit_res.points

            # กรอง irrelevant hits
            valid_vit = [h for h in vit_hits if h.score >= self.cfg.image_distance_threshold]
            if not valid_vit:
                valid_vit = vit_hits[:3] if vit_hits else []

            if not valid_vit:
                return None, False

            top_hit = valid_vit[0]

            # ตรวจ series
            if top_hit.score >= self.cfg.vit_confidence_threshold:
                raw_series = top_hit.payload.get("series", "")
                m = re.search(r'\d+', raw_series or "")
                vit_series = m.group() if m else None

            for rank, hit in enumerate(valid_vit):
                pid = hit.payload.get("product_id", "")
                if not pid:
                    continue
                if pid not in score_map:
                    score_map[pid] = {
                        "text_ranks": [], "vit_rank": None,
                        "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                    }
                if score_map[pid]["vit_rank"] is None:
                    score_map[pid]["vit_rank"] = rank

            # ViT dominant → score สูงมาก → skip text + SigLIP
            if top_hit.score >= 0.85:
                return vit_series, True

        # ── SigLIP (รันเฉพาะเมื่อ ViT ไม่ dominant) ──────
        siglip_vec = self._embed_siglip(image)
        if siglip_vec:
            qdrant_filter = None
            if vit_series:
                qdrant_filter = Filter(
                    must=[FieldCondition(
                        key="series",
                        match=MatchValue(value=f"{vit_series} Series"),
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

            for rank, hit in enumerate(_siglip_res.points):
                pid = hit.payload.get("product_id", "")
                if not pid:
                    continue
                if pid not in score_map:
                    score_map[pid] = {
                        "text_ranks": [], "vit_rank": None,
                        "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                    }
                if score_map[pid]["siglip_rank"] is None:
                    score_map[pid]["siglip_rank"] = rank

        return vit_series, False

    def _text_to_image_search(self, text: str, score_map: dict):
        """
        SigLIP text→image search สำหรับ query_type == "visual"
        (user อธิบายรูปร่างสินค้าด้วยคำพูด ไม่มีรูปจริง)
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
            if hit.score < 0.20:
                continue
            if pid not in score_map:
                score_map[pid] = {
                    "text_ranks": [], "vit_rank": None,
                    "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                }
            if score_map[pid]["siglip_rank"] is None:
                score_map[pid]["siglip_rank"] = rank

    # ──────────────────────────────────────────────────────
    # RRF FUSION (รองรับ graph signal เพิ่ม)
    # ──────────────────────────────────────────────────────

    def _rrf_fusion(self, score_map: dict) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion รวม signals:
          text, vit, siglip, keyword, graph

        RRF = Σ weight_i / (k + rank_i)
        graph signal ใช้ graph_rank แทน rank เพื่อให้ weight_graph ทำงานถูก
        """
        k = self.cfg.rrf_k
        final_scores: dict[str, float] = {}

        for pid, scores in score_map.items():
            rrf = 0.0

            # Text
            text_ranks = scores.get("text_ranks", [])
            if text_ranks:
                rrf += sum(
                    self.cfg.weight_text / (k + r) for r in text_ranks
                ) / len(text_ranks)

            # ViT
            vit_rank = scores.get("vit_rank")
            if vit_rank is not None:
                rrf += self.cfg.weight_vit / (k + vit_rank)

            # SigLIP
            siglip_rank = scores.get("siglip_rank")
            if siglip_rank is not None:
                rrf += self.cfg.weight_siglip / (k + siglip_rank)

            # Keyword (flat bonus)
            keyword = scores.get("keyword", 0.0)
            if keyword > 0:
                rrf += self.cfg.weight_keyword * keyword / 10.0

            # Graph (flat bonus — rank แปลงเป็น score แบบ RRF-like)
            graph_rank = scores.get("graph", 0.0)
            if graph_rank == 0.0 and "graph" in scores:
                # rank=0 หมายถึง seed product ที่อยู่ใน graph context
                rrf += self.cfg.weight_graph / (k + 0)
            elif isinstance(graph_rank, (int, float)) and graph_rank > 0:
                rrf += self.cfg.weight_graph / (k + graph_rank)

            final_scores[pid] = rrf

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # ──────────────────────────────────────────────────────
    # EMBEDDING HELPERS
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed_text(self, text: str) -> Optional[list[float]]:
        try:
            enc = self.jina_tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512
            ).to(self.cfg.device)
            out   = self.jina_model(**enc, return_dict=True)
            last  = out.last_hidden_state
            mask  = enc["attention_mask"].unsqueeze(-1).float()
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
        try:
            inputs = self.siglip_processor(
                text=[text], return_tensors="pt",
                padding=True, truncation=True, max_length=64
            ).to(self.cfg.device)
            feat = self.siglip_model.get_text_features(**inputs)
            feat = F.normalize(feat, dim=-1)
            return feat.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ SigLIP text embed error: {e}")
            return None

    # ──────────────────────────────────────────────────────
    # FETCH CHUNK TEXTS
    # ──────────────────────────────────────────────────────

    def _fetch_chunk_texts(self, product_id: str) -> list[dict]:
        """ดึง chunk texts ทั้งหมดของ product จาก Qdrant"""
        try:
            hits, _ = self.client.scroll(
                collection_name=self.cfg.text_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="product_id",
                        match=MatchValue(value=product_id),
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