"""
reranker.py
===========
Phase 4: Cross-Encoder Reranker สำหรับ Movex Product Chatbot

รับ RetrievalResult จาก retriever.py แล้ว:
  1. สร้างคู่ (query, product_text) ต่อทุก product
  2. ส่งเข้า cross-encoder — อ่านทั้งคู่พร้อมกัน ให้ score แม่นกว่า vector search
  3. Combine cross-encoder score กับ RRF score (weighted)
  4. คืน RerankedResult เรียงใหม่

Cross-encoder ต่างจาก bi-encoder (Jina) ตรงที่:
  - Bi-encoder: embed query และ doc แยกกัน → dot product
  - Cross-encoder: อ่าน [query, doc] พร้อมกัน → attention ไหลข้ามสองฝั่ง → แม่นกว่ามาก
  แต่ช้ากว่า จึงใช้หลัง retrieval เพื่อ rerank top-K เท่านั้น

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - เบา (22M params), เร็วบน CPU
  - เทรนบน MS MARCO passage ranking
  - เหมาะกับ product Q&A

ใช้งาน:
  from reranker import Reranker, RerankerConfig
  reranker = Reranker(cfg)
  reranked = reranker.rerank(query, retrieval_result)
"""

import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from retriever import RetrievalResult, RetrievedProduct


# ===============================================================
# DATA CLASSES
# ===============================================================

@dataclass
class RerankedProduct:
    """Product หลัง rerank พร้อม scores ทุกชั้น"""
    product_id: str
    final_score: float          # combined score (cross-encoder + rrf)
    cross_encoder_score: float  # raw cross-encoder logit
    rrf_score: float            # score จาก retriever
    matched_chunks: list[dict] = field(default_factory=list)
    matched_images: list[dict] = field(default_factory=list)
    series: str = ""
    product_type: str = ""
    material: str = ""
    keyword_boost: float = 0.0
    # chunk ที่ได้คะแนนสูงสุดจาก cross-encoder (ใช้ส่งให้ LLM)
    best_chunk_text: str = ""


@dataclass
class RerankedResult:
    """ผลลัพธ์ทั้งหมดหลัง rerank"""
    products: list[RerankedProduct]
    query_used: str
    image_search_used: bool = False
    vit_series_filter: Optional[str] = None
    total_candidates: int = 0

    def top(self, k: int = 5) -> list[RerankedProduct]:
        return self.products[:k]

    def to_llm_context(self, k: int = 3) -> str:
        """
        แปลง top-k products เป็น context string สำหรับส่งให้ LLM
        รวม chunk ที่ดีที่สุดของแต่ละ product
        """
        parts = []
        for i, p in enumerate(self.products[:k], 1):
            parts.append(
                f"[Product {i}] {p.product_id} (score: {p.final_score:.3f})\n"
                f"{p.best_chunk_text}"
            )
        return "\n\n---\n\n".join(parts)


# ===============================================================
# RERANKER CONFIG
# ===============================================================

@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # จำนวน products ที่จะ rerank (รับมาจาก retriever)
    rerank_top_k: int = 10

    # จำนวน products ที่จะคืนหลัง rerank
    final_top_k: int = 5

    # max token ต่อ input pair (query + product text)
    max_length: int = 512

    # Weight สำหรับ combine scores
    # final = w_ce * norm(cross_encoder) + w_rrf * norm(rrf)
    weight_cross_encoder: float = 0.7
    weight_rrf: float = 0.3

    # Keyword boost ยังคงทำงาน — ถ้า product มี keyword boost สูง ให้ priority
    keyword_boost_bonus: float = 0.15

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# RERANKER
# ===============================================================

class Reranker:
    """
    Cross-Encoder Reranker

    รับ RetrievalResult จาก Retriever แล้ว rerank ด้วย cross-encoder
    คืน RerankedResult พร้อม context พร้อมส่งให้ LLM

    Parameters
    ----------
    cfg : RerankerConfig
    """

    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        print(f"⏳ Loading cross-encoder: {cfg.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self.model.eval().to(cfg.device)
        print(f"✅ Cross-encoder loaded ({cfg.device})")

    # -----------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------

    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> RerankedResult:
        """
        รับ query + RetrievalResult แล้วคืน RerankedResult

        Parameters
        ----------
        query : str
            original user query (ใช้เป็นฝั่งซ้ายของ cross-encoder)
        retrieval_result : RetrievalResult
            ผลจาก Retriever.search()
        """
        candidates = retrieval_result.products[:self.cfg.rerank_top_k]

        if not candidates:
            return RerankedResult(
                products=[],
                query_used=query,
                image_search_used=retrieval_result.image_search_used,
                vit_series_filter=retrieval_result.vit_series_filter,
                total_candidates=0,
            )

        # ── 1. Score แต่ละ product ด้วย cross-encoder ─────────
        ce_scores = []  # (product_idx, chunk_idx, score)
        best_scores = {}  # product_idx → (best_score, best_chunk_text)

        for p_idx, product in enumerate(candidates):
            chunks = product.matched_chunks
            if not chunks:
                # ถ้าไม่มี chunk ให้ใช้ product_id แทน
                chunks = [{"text": product.product_id, "type": "id"}]

            # score ทุก chunk ของ product นี้
            chunk_scores = self._score_chunks(query, chunks)

            # เก็บ best chunk ของแต่ละ product
            best_idx = max(range(len(chunk_scores)), key=lambda i: chunk_scores[i])
            best_scores[p_idx] = (
                chunk_scores[best_idx],
                chunks[best_idx].get("text", ""),
            )
            ce_scores.append(chunk_scores)

        # ── 2. Normalize scores ────────────────────────────────
        raw_ce   = [best_scores[i][0] for i in range(len(candidates))]
        raw_rrf  = [p.rrf_score for p in candidates]

        norm_ce  = self._minmax_normalize(raw_ce)
        norm_rrf = self._minmax_normalize(raw_rrf)

        # ── 3. Combine + Keyword Bonus ─────────────────────────
        final_scores = []
        for i, product in enumerate(candidates):
            score = (
                self.cfg.weight_cross_encoder * norm_ce[i]
                + self.cfg.weight_rrf * norm_rrf[i]
            )
            # keyword boost เพิ่มเป็น bonus ตรงๆ
            if product.keyword_boost > 0:
                score += self.cfg.keyword_boost_bonus
            final_scores.append(score)

        # ── 4. สร้าง RerankedProduct list ─────────────────────
        reranked = []
        for i, product in enumerate(candidates):
            best_ce_score, best_text = best_scores[i]
            reranked.append(RerankedProduct(
                product_id=product.product_id,
                final_score=final_scores[i],
                cross_encoder_score=best_ce_score,
                rrf_score=product.rrf_score,
                matched_chunks=product.matched_chunks,
                matched_images=product.matched_images,
                series=product.series,
                product_type=product.product_type,
                material=product.material,
                keyword_boost=product.keyword_boost,
                best_chunk_text=best_text,
            ))

        # ── 5. เรียงใหม่ตาม final_score ───────────────────────
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        return RerankedResult(
            products=reranked[:self.cfg.final_top_k],
            query_used=query,
            image_search_used=retrieval_result.image_search_used,
            vit_series_filter=retrieval_result.vit_series_filter,
            total_candidates=len(candidates),
        )

    # -----------------------------------------------------------
    # CROSS-ENCODER SCORING
    # -----------------------------------------------------------

    @torch.no_grad()
    def _score_chunks(self, query: str, chunks: list[dict]) -> list[float]:
        """
        Score ทุก chunk ของ product หนึ่งๆ กับ query

        Cross-encoder อ่าน [CLS] query [SEP] chunk_text [SEP]
        แล้ว output logit → สูง = relevant มาก
        """
        scores = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                scores.append(-999.0)
                continue

            try:
                inputs = self.tokenizer(
                    query, text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.max_length,
                    padding=True,
                ).to(self.cfg.device)

                logits = self.model(**inputs).logits
                # ms-marco model output เดียว (relevance score)
                score = logits.squeeze().item()
                scores.append(score)
            except Exception as e:
                print(f"⚠️ Cross-encoder error: {e}")
                scores.append(-999.0)

        return scores

    # -----------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------

    @staticmethod
    def _minmax_normalize(values: list[float]) -> list[float]:
        """Normalize list of floats to [0, 1]"""
        if not values:
            return values
        mn, mx = min(values), max(values)
        if mx == mn:
            return [1.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]


# ===============================================================
# CONVENIENCE FUNCTION
# ===============================================================

def load_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: Optional[str] = None,
    final_top_k: int = 5,
) -> Reranker:
    """Helper สำหรับโหลด reranker ด้วยค่า default"""
    cfg = RerankerConfig(
        model_name=model_name,
        final_top_k=final_top_k,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return Reranker(cfg)