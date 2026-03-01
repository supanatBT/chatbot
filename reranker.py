"""
reranker.py
===========
Phase 4: Cross-Encoder Reranker สำหรับ Movex Product Chatbot

แก้จาก v3:
  ✦ ลบ import จาก retriever.py เก่า
  ✦ รับ retrieval_result แบบ duck-typing (ไม่ผูกกับ class ใด)
  ✦ RerankedProduct fields ครบเหมือนเดิมทุกอย่าง
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


# ===============================================================
# DATA CLASSES
# ===============================================================

@dataclass
class RerankedProduct:
    """Product หลัง rerank พร้อม scores ทุกชั้น"""
    product_id:          str
    final_score:         float
    cross_encoder_score: float
    rrf_score:           float
    matched_chunks:      list = field(default_factory=list)
    matched_images:      list = field(default_factory=list)
    series:              str   = ""
    product_type:        str   = ""
    material:            str   = ""
    keyword_boost:       float = 0.0
    best_chunk_text:     str   = ""


@dataclass
class RerankedResult:
    """ผลลัพธ์ทั้งหมดหลัง rerank"""
    products:           list
    query_used:         str
    image_search_used:  bool = False
    vit_series_filter:  Optional[str] = None
    total_candidates:   int  = 0
    # v4 compat — graph_context/graph_result passthrough จาก retrieval_result
    graph_context:      object = None
    graph_result:       object = None

    def top(self, k: int = 5) -> list:
        return self.products[:k]

    def to_llm_context(self, k: int = 3) -> str:
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
    model_name:           str   = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_top_k:         int   = 10
    final_top_k:          int   = 5
    max_length:           int   = 512
    weight_cross_encoder: float = 0.7
    weight_rrf:           float = 0.3
    keyword_boost_bonus:  float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# RERANKER
# ===============================================================

class Reranker:
    """
    Cross-Encoder Reranker

    รับ retrieval_result จาก RetrieverV4 (หรือ V3) แบบ duck-typing
    — ต้องมี .products (list ที่มี product_id, rrf_score, matched_chunks,
      matched_images, series, product_type, material, keyword_boost)
    — และ .image_search_used, .vit_series_filter (optional)
    """

    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        print(f"⏳ Loading cross-encoder: {cfg.model_name}...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self.model.eval().to(cfg.device)
        print(f"✅ Cross-encoder loaded ({cfg.device})")

    # -----------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------

    def rerank(self, query: str, retrieval_result) -> RerankedResult:
        candidates = retrieval_result.products[:self.cfg.rerank_top_k]

        if not candidates:
            return RerankedResult(
                products=[],
                query_used=query,
                image_search_used=getattr(retrieval_result, 'image_search_used', False),
                vit_series_filter=getattr(retrieval_result, 'vit_series_filter', None),
                graph_context=getattr(retrieval_result, 'graph_context', None),
                graph_result=getattr(retrieval_result, 'graph_result', None),
                total_candidates=0,
            )
        # ── 0. ตรวจว่าควร skip CE ไหม ──────────────────────── ← เพิ่มใหม่
        vit_signal = getattr(retrieval_result, 'vit_signal', None)
        image_used = getattr(retrieval_result, 'image_search_used', False)
        skip_ce = (
            image_used and
            vit_signal is not None and
            vit_signal.confidence >= 0.50
        )
        self._last_skip_ce = skip_ce
        # ── 1. Score แต่ละ product ด้วย cross-encoder ─────────
        best_scores = {}

        for p_idx, product in enumerate(candidates):
            chunks = getattr(product, 'matched_chunks', []) or []
            if not chunks:
                chunks = [{"text": product.product_id, "type": "id"}]

            if skip_ce:                                          # ← เพิ่มใหม่
                best_scores[p_idx] = (0.0, chunks[0].get("text", ""))
            else:
                chunk_scores = self._score_chunks(query, chunks)
                best_idx = max(range(len(chunk_scores)), key=lambda i: chunk_scores[i])
                best_scores[p_idx] = (
                    chunk_scores[best_idx],
                    chunks[best_idx].get("text", ""),
                )

        # ── 2. Normalize ───────────────────────────────────────
        raw_ce  = [best_scores[i][0] for i in range(len(candidates))]
        raw_rrf = [p.rrf_score for p in candidates]

        norm_ce  = self._minmax_normalize(raw_ce)
        norm_rrf = self._minmax_normalize(raw_rrf)

        # ── 3. Combine + Keyword Bonus ─────────────────────────
        final_scores = []
        for i, product in enumerate(candidates):
            if skip_ce:                                          # ← เพิ่มใหม่
                score = norm_rrf[i]
            else:
                score = (
                    self.cfg.weight_cross_encoder * norm_ce[i]
                    + self.cfg.weight_rrf * norm_rrf[i]
                )
            if getattr(product, 'keyword_boost', 0.0) > 0:
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
                matched_chunks=getattr(product, 'matched_chunks', []),
                matched_images=getattr(product, 'matched_images', []),
                series=getattr(product, 'series', ''),
                product_type=getattr(product, 'product_type', ''),
                material=getattr(product, 'material', ''),
                keyword_boost=getattr(product, 'keyword_boost', 0.0),
                best_chunk_text=best_text,
            ))

        reranked.sort(key=lambda x: x.final_score, reverse=True)

        return RerankedResult(
            products=reranked[:self.cfg.final_top_k],
            query_used=query,
            image_search_used=getattr(retrieval_result, 'image_search_used', False),
            vit_series_filter=getattr(retrieval_result, 'vit_series_filter', None),
            graph_context=getattr(retrieval_result, 'graph_context', None),
            graph_result=getattr(retrieval_result, 'graph_result', None),
            total_candidates=len(candidates),
        )
    # -----------------------------------------------------------
    # CROSS-ENCODER SCORING
    # -----------------------------------------------------------

    @torch.no_grad()
    def _score_chunks(self, query: str, chunks: list) -> list:
        scores = []
        for chunk in chunks:
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
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
                scores.append(logits.squeeze().item())
            except Exception as e:
                print(f"⚠️ Cross-encoder error: {e}")
                scores.append(-999.0)
        return scores

    # -----------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------

    @staticmethod
    def _minmax_normalize(values: list) -> list:
        if not values:
            return values
        mn, mx = min(values), max(values)
        if mx == mn:
            return [1.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]


# ===============================================================
# CONVENIENCE
# ===============================================================

def load_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: Optional[str] = None,
    final_top_k: int = 5,
) -> Reranker:
    cfg = RerankerConfig(
        model_name=model_name,
        final_top_k=final_top_k,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return Reranker(cfg)
