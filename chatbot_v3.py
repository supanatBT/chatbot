"""
chatbot_v3.py
=============
Phase 5 (v3): Agentic Generation — Full Pipeline with GraphRAG + Full Logging

Pipeline v3:
  User Query / Image
       ↓
  RetrieverV3 (GraphRAG: ViT → graph expand → text + RRF)
       ↓
  Reranker (Cross-Encoder)
       ↓
  Gemini Agent (function calling + Clarification Dialog)
       ↓
  PipelineLogger (บันทึกทุก step → JSONL + CSV + TXT)
"""

import os, json, re, base64, csv, torch, time
from io import BytesIO
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import google.generativeai as genai

from transformers import (
    AutoTokenizer, AutoModel,
    ViTImageProcessor, ViTForImageClassification,
    SiglipModel, SiglipProcessor,
)
from peft import PeftModel

# v3 modules
from knowledge_graph import KnowledgeGraph
from retriever_v3 import RetrieverV3, RetrieverV3Config
from reranker import Reranker, RerankerConfig
from pipeline_logger import PipelineLogger

# ================================================================
# CONFIG
# ================================================================
# GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ ไม่พบ GEMINI_API_KEY กรุณาตรวจสอบไฟล์ .env")
QDRANT_PATH     = "./qdrant_db"
JSON_PATH       = "./extracted_data_v3_machine_ready.json"
KG_DB_PATH      = "./movex_kg.db"          # สร้างด้วย: python build_graph.py
VIT_MODEL_PATH  = "./movex-lora-tuned-3"
VIT_BASE_MODEL  = "google/vit-base-patch16-224-in21k"
VIT_NUM_LABELS  = 10
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
JINA_MODEL_ID   = "jinaai/jina-embeddings-v3"
GEMINI_MODEL    = "gemini-2.5-flash"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# STARTUP
# ================================================================
print("=" * 60)
print("  Movex Chatbot v3 — GraphRAG + Full Pipeline Logging")
print(f"  Device: {DEVICE}")
print("=" * 60)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)

product_db  = {p["product_id"]: p for p in _data.get("products", []) if "product_id" in p}
material_db = {m["material_code"]: m for m in _data.get("Materials", [])}
process_db  = {p["manufacturing_process"]: p for p in _data.get("manufacturing_process", [])}
print(f"✅ JSON: {len(product_db)} products | {len(material_db)} materials | {len(process_db)} processes")

print("⏳ โหลด Jina v3...")
jina_tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL_ID, trust_remote_code=True)
jina_model = AutoModel.from_pretrained(
    JINA_MODEL_ID, trust_remote_code=True,
    dtype=torch.float32, attn_implementation="eager"
).to(DEVICE)
jina_model.eval()
print("✅ Jina v3")

print("⏳ โหลด ViT LoRA...")
vit_processor = ViTImageProcessor.from_pretrained(VIT_BASE_MODEL)
_base_vit     = ViTForImageClassification.from_pretrained(VIT_BASE_MODEL, num_labels=VIT_NUM_LABELS)
vit_model     = PeftModel.from_pretrained(_base_vit, VIT_MODEL_PATH)
vit_model.eval().to(DEVICE)
print("✅ ViT LoRA")

print("⏳ โหลด SigLIP...")
siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_ID)
siglip_model     = SiglipModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE)
siglip_model.eval()
print("✅ SigLIP")

kg = KnowledgeGraph.load_from_db(KG_DB_PATH)

_retriever_inner = RetrieverV3(
    cfg=RetrieverV3Config(qdrant_path=QDRANT_PATH, device=DEVICE),
    jina_tokenizer=jina_tokenizer,
    jina_model=jina_model,
    vit_processor=vit_processor,
    vit_model=vit_model,
    siglip_processor=siglip_processor,
    siglip_model=siglip_model,
    product_db=product_db,
    kg=kg,
)
print("✅ RetrieverV3 + KnowledgeGraph")

reranker = Reranker(RerankerConfig(final_top_k=5))
print("✅ Reranker")

genai.configure(api_key=GEMINI_API_KEY)
print("✅ Gemini configured")

logger = PipelineLogger(log_dir="./logs")
print(f"✅ PipelineLogger → {logger.log_paths()}")
print("=" * 60)


# ================================================================
# HELPERS
# ================================================================

def image_to_base64_html(pil_image):
    if pil_image is None:
        return ""
    buf = BytesIO()
    pil_image.thumbnail((400, 400))
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/jpeg;base64,{b64}" style="max-width:250px;border-radius:10px;margin-bottom:10px;" />'


def _enrich_product(p: dict) -> dict:
    p = p.copy()
    mat  = p.get("Material", "")
    proc = p.get("manufacturing_process", "")
    if mat in material_db:
        p["material_details"] = material_db[mat]
    if proc in process_db:
        p["manufacturing_details"] = process_db[proc]
    return p


def _build_graph_summary(ctx) -> str:
    """
    สร้าง graph summary สำหรับส่งให้ LLM

    สำคัญ: ต้องบอกชัดว่า process/material นั้นเป็นของ seed product ตัวไหน
    เพราะ top1 ใน RAG context อาจไม่ใช่ seed product (drift จาก text search)
    LLM จะได้ไม่สับสนและตอบได้ถูกต้อง
    """
    if not ctx:
        return "ไม่มี graph context"

    parts = []

    # บอก seed products ก่อนเสมอ — LLM จะได้รู้ว่า context นี้เป็นของใคร
    if ctx.seed_products:
        parts.append(f"[Seed products ที่ระบบ detect ได้]: {', '.join(ctx.seed_products)}")

    # Compatible products
    if ctx.compatible_products:
        parts.append(
            f"[Compatible parts สำหรับ {', '.join(ctx.seed_products)}]: "
            f"{', '.join(ctx.compatible_products[:5])}"
        )

    # Same series
    if ctx.same_series:
        parts.append(
            f"[สินค้า series เดียวกัน]: {', '.join(ctx.same_series[:5])}"
        )

    # Material — ระบุว่าเป็นของ seed ตัวไหน
    if ctx.material_details:
        for mat, info in ctx.material_details.items():
            full = info.get("full_name", mat)
            desc = info.get("description", "")
            best = info.get("best_used_for", "")
            line = (
                f"[วัสดุของ {', '.join(ctx.seed_products)}]: "
                f"รหัส={mat}, ชื่อ={full}"
            )
            if desc:
                line += f", คุณสมบัติ={desc[:120]}"
            if best:
                line += f", เหมาะกับ={best[:80]}"
            parts.append(line)

    # Process — ระบุว่าเป็นของ seed ตัวไหน พร้อมรายละเอียดครบ
    if ctx.process_details:
        for proc, info in ctx.process_details.items():
            full = info.get("full_name", proc)
            desc = info.get("description", "")
            adv  = info.get("advantages", "") or info.get("best_used_for", "")
            line = (
                f"[กระบวนการผลิตของ {', '.join(ctx.seed_products)}]: "
                f"วิธี={full}"
            )
            if desc:
                line += f", รายละเอียด={desc[:150]}"
            if adv:
                line += f", ข้อดี={adv[:100]}"
            parts.append(line)

    return "\n".join(parts) if parts else "ไม่มี graph context"


# ================================================================
# INSTRUMENTED RETRIEVER — ดัก raw hits จากทุก step
# ================================================================

class InstrumentedRetriever:
    """
    Wrapper รอบ RetrieverV3 — monkey-patch internal methods
    เพื่อดักจับ raw hits ก่อนผ่าน RRF สำหรับส่งให้ PipelineLogger

    สิ่งที่ capture:
      _vit_hits_raw      — Qdrant ScoredPoints จาก ViT (ก่อน filter)
      _siglip_hits_raw   — Qdrant ScoredPoints จาก SigLIP
      _vit_top_score     — score ของ hit อันดับ 1 จาก ViT
      _vit_series        — series ที่ ViT detect ได้
      _vit_dominant      — True ถ้า ViT score >= 0.85
      _text_hits_raw     — Qdrant ScoredPoints จาก Jina text search
      _text_skipped      — True ถ้า skip เพราะ vit_dominant
      _score_map_snapshot — score_map หลังทุก signal รวมกันก่อน RRF
      _ranked_snapshot   — [(pid, rrf_score)] หลัง RRF
      _graph_triggered_by — "vit" | "keyword" | "none"
    """

    def __init__(self, inner: RetrieverV3):
        self.inner = inner
        self._reset()
        self._patch()

    def _reset(self):
        self._vit_hits_raw:      list = []
        self._siglip_hits_raw:   list = []
        self._vit_top_score:     float = 0.0
        self._vit_series:        str = ""
        self._vit_dominant:      bool = False
        self._text_hits_raw:     list = []
        self._text_skipped:      bool = False
        self._text_skip_reason:  str = ""
        self._score_map_snapshot: dict = {}
        self._ranked_snapshot:   list = []
        self._graph_triggered_by: str = "none"

    def _patch(self):
        r = self.inner
        self._orig_img   = r._image_search
        self._orig_txt   = r._text_search
        self._orig_kw    = r._keyword_boost
        self._orig_gvit  = r._graph_expand_from_vit
        self._orig_gkw   = r._graph_expand_from_keywords
        self._orig_rrf   = r._rrf_fusion

        r._image_search               = self._h_image
        r._text_search                = self._h_text
        r._keyword_boost              = self._h_kw
        r._graph_expand_from_vit      = self._h_gvit
        r._graph_expand_from_keywords = self._h_gkw
        r._rrf_fusion                 = self._h_rrf

    def search(self, *a, **kw):
        self._reset()
        return self.inner.search(*a, **kw)

    # ── Hooks ───────────────────────────────────────────────

    def _h_kw(self, query, score_map, vit_dominant: bool = False):
        self._orig_kw(query, score_map, vit_dominant=vit_dominant)

    def _h_image(self, image, score_map):
        """Replicate _image_search + capture raw hits"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        cfg = self.inner.cfg
        image = image.convert("RGB")
        vit_vec  = self.inner._embed_vit(image)
        vit_hits = []
        vit_series = None
        vit_top_score = 0.0

        if vit_vec:
            res = self.inner.client.query_points(
                collection_name=cfg.image_collection,
                query=vit_vec, using="vit",
                limit=cfg.image_candidates, with_payload=True,
            )
            vit_hits = res.points
            valid = [h for h in vit_hits if h.score >= cfg.image_distance_threshold]
            if not valid:
                valid = vit_hits[:3] if vit_hits else []
            if valid:
                top = valid[0]
                vit_top_score = top.score
                if top.score >= cfg.vit_confidence_threshold:
                    m = re.search(r'\d+', top.payload.get("series", "") or "")
                    vit_series = m.group() if m else None
                for rank, h in enumerate(valid):
                    pid = h.payload.get("product_id", "")
                    if not pid:
                        continue
                    score_map.setdefault(pid, {
                        "text_ranks": [], "vit_rank": None,
                        "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                    })
                    if score_map[pid]["vit_rank"] is None:
                        score_map[pid]["vit_rank"] = rank
                if top.score >= 0.85:
                    self._vit_hits_raw    = vit_hits
                    self._siglip_hits_raw = []
                    self._vit_top_score   = vit_top_score
                    self._vit_series      = vit_series or ""
                    self._vit_dominant    = True
                    return vit_series, True

        siglip_vec = self.inner._embed_siglip(image)
        sig_hits   = []
        if siglip_vec:
            filt = None
            if vit_series:
                filt = Filter(must=[FieldCondition(
                    key="series", match=MatchValue(value=f"{vit_series} Series")
                )])
            res2 = self.inner.client.query_points(
                collection_name=cfg.image_collection,
                query=siglip_vec, using="siglip",
                limit=cfg.image_candidates, query_filter=filt, with_payload=True,
            )
            sig_hits = res2.points
            for rank, h in enumerate(sig_hits):
                pid = h.payload.get("product_id", "")
                if not pid:
                    continue
                score_map.setdefault(pid, {
                    "text_ranks": [], "vit_rank": None,
                    "siglip_rank": None, "keyword": 0.0, "graph": 0.0
                })
                if score_map[pid]["siglip_rank"] is None:
                    score_map[pid]["siglip_rank"] = rank

        self._vit_hits_raw    = vit_hits
        self._siglip_hits_raw = sig_hits
        self._vit_top_score   = vit_top_score
        self._vit_series      = vit_series or ""
        self._vit_dominant    = False
        return vit_series, False

    def _h_text(self, query, score_map):
        """Replicate _text_search + capture"""
        vec = self.inner._embed_text(query)
        if vec is None:
            self._text_skipped = True
            self._text_skip_reason = "embed_failed"
            return
        res = self.inner.client.query_points(
            collection_name=self.inner.cfg.text_collection,
            query=vec, using="jina",
            limit=self.inner.cfg.text_candidates, with_payload=True,
        )
        self._text_hits_raw = res.points
        for rank, h in enumerate(res.points):
            pid = h.payload.get("product_id", "")
            if not pid:
                continue
            score_map.setdefault(pid, {
                "text_ranks": [], "vit_rank": None,
                "siglip_rank": None, "keyword": 0.0, "graph": 0.0
            })
            score_map[pid]["text_ranks"].append(rank)

    def _h_gvit(self, vit_series, score_map):
        self._graph_triggered_by = "vit"
        return self._orig_gvit(vit_series, score_map)

    def _h_gkw(self, query, score_map):
        kw_hits = [p for p, s in score_map.items() if s.get("keyword", 0) > 0]
        if kw_hits:
            self._graph_triggered_by = "keyword"
        return self._orig_gkw(query, score_map)

    def _h_rrf(self, score_map):
        """Snapshot score_map ก่อน RRF แล้ว return ranked"""
        self._score_map_snapshot = {k: dict(v) for k, v in score_map.items()}
        ranked = self._orig_rrf(score_map)
        self._ranked_snapshot = ranked
        return ranked


retriever = InstrumentedRetriever(_retriever_inner)


# ================================================================
# TOOL CALL TRACKER
# ================================================================

_tool_call_log: list[dict] = []

def _tracked(fn):
    """Decorator: log tool call args + result"""
    import inspect
    params = list(inspect.signature(fn).parameters.keys())
    def wrapper(*args, **kwargs):
        entry = {"tool_name": fn.__name__, "args": {}, "result": "", "success": True}
        try:
            entry["args"] = {**dict(zip(params, args)), **kwargs}
            result = fn(*args, **kwargs)
            entry["result"] = result
            return result
        except Exception as e:
            entry["success"] = False
            entry["result"]  = str(e)
            raise
        finally:
            _tool_call_log.append(entry)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__  = fn.__doc__
    return wrapper

# ================================================================
# AI TOOLS
# ================================================================

def analyze_product_database(product_category: str = "", keyword: str = "") -> str:
    """Tool: วิเคราะห์และนับจำนวนสินค้าภาพรวม"""
    cat_lower = product_category.lower()
    kw_lower  = keyword.lower()
    matched   = []
    for pid, p in product_db.items():
        all_text = " ".join(
            str(v).lower() for v in p.values() if isinstance(v, (str, int, float))
        ) + f" {pid.lower()}"
        if cat_lower and cat_lower not in all_text:
            continue
        if kw_lower and not all(kw in all_text for kw in kw_lower.split()):
            continue
        matched.append(pid)
    if not matched:
        return "ระบบหาข้อมูลไม่พบ"
    return f"พบสินค้าที่ตรงเงื่อนไขทั้งหมด {len(matched)} รุ่น ได้แก่: {', '.join(matched)}"


def compare_specific_products(product_a: str, product_b: str) -> str:
    """Tool: เปรียบเทียบสินค้า 2 รุ่น"""
    def find(target):
        clean = re.sub(r'[^a-zA-Z0-9]', '', target).lower()
        for pid, p in product_db.items():
            if clean in re.sub(r'[^a-zA-Z0-9]', '', pid).lower():
                return _enrich_product(p)
        return {"error": f"ไม่พบข้อมูลรุ่น {target}"}
    return json.dumps([find(product_a), find(product_b)], ensure_ascii=False)


def filter_product_specs(spec_type: str, condition: str, target_value: float) -> str:
    """Tool: กรองสินค้าตาม spec และเงื่อนไขตัวเลข"""
    key_map = {
        "width":  ["Plate_Width_mm"],
        "weight": ["Weight_kg_m"],
        "pitch":  ["Pitch_mm"],
        "radius": ["Radius_min_mm", "Min_curve_radius_mm"],
        "load":   ["Max_Working_Load_N"],
        "teeth":  ["Z_Teeth"],
        "bore":   ["Bore_mm"],
        "pd":     ["PD_mm"],
        "od":     ["OD_mm"],
        "s":      ["S_mm"],
    }
    target_keys = key_map.get(spec_type.lower(), [spec_type])
    matched = []
    for pid, p in product_db.items():
        val = None
        for key in target_keys:
            if key in p and p[key] is not None:
                val = float(p[key]) if isinstance(p[key], (int, float)) else None
                if val is None:
                    nums = re.findall(r'\d+\.?\d*', str(p[key]))
                    if nums:
                        val = float(nums[0])
                if val is not None:
                    break
        if val is None and spec_type.lower() == "width":
            km = re.search(r'K(\d{3,4})', pid)
            if km:
                val = (float(km.group(1)) / 100) * 25.4
        if val is not None:
            ok = (condition == "<"  and val <  target_value or
                  condition == "<=" and val <= target_value or
                  condition == ">"  and val >  target_value or
                  condition == ">=" and val >= target_value or
                  condition == "==" and val == target_value)
            if ok:
                matched.append(f"{pid} ({val})")
    if not matched:
        return f"ไม่พบสินค้าที่มี {spec_type} {condition} {target_value}"
    return f"พบ {len(matched)} รุ่น ได้แก่: {', '.join(matched)}"

# Re-wrap tracked versions
tracked_analyze    = _tracked(analyze_product_database)
tracked_compare    = _tracked(compare_specific_products)
tracked_filter     = _tracked(filter_product_specs)


# ================================================================
# SYSTEM PROMPT
# ================================================================

def _build_context_tag(reranked_result, graph_ctx, ai_reply: str = "") -> str:
    """
    สร้าง context summary สั้นๆ สำหรับติดไปกับ history แต่ละ turn
    ทำให้ Gemini รู้ว่า turn นั้นๆ มีข้อมูล product อะไรอยู่
    รวม sprocket ที่ LLM พูดถึงใน reply ด้วย เพื่อให้ turn ถัดไปหา process ได้ถูก
    """
    lines = []
    if reranked_result and reranked_result.products:
        top = reranked_result.products[0]
        lines.append(f"top_product={top.product_id}")
    if graph_ctx:
        if graph_ctx.seed_products:
            lines.append(f"seeds={','.join(graph_ctx.seed_products)}")
        if graph_ctx.compatible_products:
            lines.append(f"compatible={','.join(graph_ctx.compatible_products)}")
            # เก็บ sprocket แยกไว้ให้ชัด — ใช้ตอน turn ถัดไปถามว่า "สเตอร์นั้นผลิตยังไง"
            sprockets = [p for p in graph_ctx.compatible_products if "sprocket" in p.lower()]
            if sprockets:
                lines.append(f"sprockets={','.join(sprockets)}")
        if graph_ctx.process_details:
            lines.append(f"process={','.join(graph_ctx.process_details.keys())}")
        if graph_ctx.material_details:
            lines.append(f"material={','.join(graph_ctx.material_details.keys())}")
    return " | ".join(lines)


def _build_system_prompt(json_context: str, graph_summary: str) -> str:
    return f"""
คุณคือ 'Movex Assistant' วิศวกรฝ่ายขายมืออาชีพของบริษัท Movex
ตอบภาษาไทย เป็นธรรมชาติ มีหางเสียง (ครับ/ค่ะ)

[RAG Context — สินค้าที่ระบบค้นหามาได้ เรียงตาม final_score สูงสุด]
{json_context}

[GraphRAG Context — ความสัมพันธ์ของสินค้า]
{graph_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 ขั้นตอนที่ 1 (บังคับ): ตรวจก่อนตอบทุกครั้ง
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ก่อนตอบทุกครั้ง ให้ถามตัวเองว่า "ลูกค้าระบุรุ่นสินค้าชัดเจนแล้วหรือยัง?"

✅ ระบุชัด = มีอย่างน้อย 1 ข้อต่อไปนี้:
  • รหัสรุ่น / Ref ชัดเจน เช่น "LF820 K325", "54901"
  • series + ความกว้าง เช่น "880 กว้าง 82.5"
  • รูปภาพของสินค้าชัดเจน (ระบบ ViT detect series ได้)

❌ ยังไม่ระบุ = ถามแบบนี้เท่านั้น:
  • "มีสายพานเลี้ยวได้ไหม"   ← ไม่มี series ไม่มี width
  • "แนะนำสายพานหน่อย"       ← ไม่มีข้อมูลใดเลย
  • "สายพาน SS มีอะไรบ้าง"   ← ไม่ระบุ type/width
  • "อยากเปลี่ยนสายพาน"      ← ไม่รู้รุ่นเดิม

ถ้า ❌ → ต้องถามกลับก่อนเสมอ ห้ามดู spec ห้ามตอบรุ่นใดรุ่นหนึ่ง

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 ขั้นตอนที่ 2: วิธีถามกลับ (Clarification)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ถามเฉพาะ 2-3 ข้อที่จำเป็น พร้อมตัวเลือกให้เลือก:

ตัวอย่าง "มีสายพานเลี้ยวได้ไหม":
  "ขอถามเพิ่มเติมก่อนนะครับ เพื่อแนะนำได้ตรงที่สุด:
  • ความกว้างที่ต้องการ: 82.5 / 101.6 / 114.3 / 190.5 mm หรืออื่นๆ?
  • รัศมีโค้งขั้นต่ำ: มีข้อกำหนดไหมครับ เช่น ≥ 400mm?
  • สินค้าที่ลำเลียง: เช่น ขวด PET / กระป๋อง / ลัง?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 ขั้นตอนที่ 3: ตอบ spec (เฉพาะเมื่อระบุรุ่นชัดแล้ว)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- ยึดข้อมูลจาก _rank 1 เป็นหลัก
- ห้ามมั่วตัวเลขหรือรหัสสินค้าเด็ดขาด
- "เลี้ยวได้ไหม" → ดู Min_curve_radius_mm / Radius_min_mm
  มีค่า = เลี้ยวได้ (บอก radius) / ไม่มีค่า = Straight Running เลี้ยวไม่ได้
- "workload" → ดู Max_Working_Load_N
- "ผลิตยังไง / กระบวนการผลิต / วัสดุ" → ดูจาก GraphRAG Context [กระบวนการผลิตของ...] และ [วัสดุของ...]
  ถ้า GraphRAG seeds มีหลายตัว → ตอบให้ครบทุกตัวที่อยู่ใน seeds
  ห้ามตอบแค่ตัวเดียวถ้า context ระบุหลายรุ่น เช่น seeds=54901,55001 → ตอบทั้ง 2
- ถ้า final_score < 0.2 ทุกรุ่น → แจ้งว่าข้อมูลไม่เพียงพอ
- ห้ามเรียก Tool เพื่อตอบ spec ที่อยู่ใน RAG Context แล้ว

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 กฎการใช้ Tool (เรียกเมื่อจำเป็นเท่านั้น)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ถามจำนวน/ภาพรวมทั้งหมด → Tool: analyze_product_database
2. ถามหา Sprocket คู่ → ดูจาก GraphRAG Context [Compatible parts] ก่อนเสมอ
   ห้ามใช้ Tool เพื่อหา Sprocket ถ้า GraphRAG Context มีข้อมูลอยู่แล้ว
3. เปรียบเทียบ 2 รุ่นขึ้นไป → Tool: compare_specific_products
4. กรองตาม spec ตัวเลข เช่น "กว้างไม่เกิน 100mm" → Tool: filter_product_specs
"""


# ================================================================
# MAIN CHAT FUNCTION
# ================================================================

def chat_interaction(
    user_text: str,
    img_input,
    chat_history: list,
    img_w: float,
    txt_w: float,
    session_id: str = "",
):
    global _tool_call_log

    if not user_text.strip() and img_input is None:
        return chat_history, chat_history, "⚠️ กรุณาพิมพ์ข้อความหรืออัปโหลดรูปภาพ", session_id

    if not session_id:
        session_id = logger.new_session()

    _tool_call_log = []
    t_start = time.perf_counter()

    display_content = ""
    if img_input:
        display_content += image_to_base64_html(img_input) + "<br>"
    if user_text:
        display_content += user_text
    if not display_content:
        display_content = "[แนบรูปภาพ]"

    raw_refs_html = "### 🔍 ข้อมูลอ้างอิง (Debug v3)\n\n"
    ai_reply        = ""
    reranked_result = None
    error_msg       = None

    effective_text = user_text or "ลูกค้าส่งรูปภาพมา"
    query_type     = "image" if img_input is not None else "general"

    # ── ดึง context จาก history turn ล่าสุด ──────────────────
    # ถ้าไม่มีรูป และ query สั้น → prepend pid จาก turn ก่อน
    # เพื่อให้ retriever keyword boost + graph expand ทำงานได้ถูก series
    if img_input is None and user_text.strip() and len(user_text.split()) <= 5:
        last_ctx = ""
        for msg in reversed(chat_history):
            if msg.get("role") == "user" and msg.get("_context"):
                last_ctx = msg["_context"]
                break
        if last_ctx:
            # ถ้า query เกี่ยวกับสเตอร์/กระบวนการผลิต → ใช้ sprocket pid เป็น seed
            sprocket_kw = any(w in user_text.lower() for w in [
                "สเตอร์", "sprocket", "ผลิต", "กระบวนการ", "วัสดุสเตอร์"
            ])
            if sprocket_kw:
                m = re.search(r'sprockets=([^\s|]+)', last_ctx)
                if m:
                    first_sprocket = m.group(1).split(",")[0].strip()
                    effective_text = f"{first_sprocket} {user_text}"
                else:
                    # fallback: ใช้ compatible ตัวแรก
                    m2 = re.search(r'compatible=([^\s|]+)', last_ctx)
                    if m2:
                        first_compat = m2.group(1).split(",")[0].strip()
                        effective_text = f"{first_compat} {user_text}"
            else:
                # query ทั่วไป → ใช้ top_product
                m = re.search(r'top_product=([^\s|]+)', last_ctx)
                if m:
                    top_pid = m.group(1).strip()
                    effective_text = f"{top_pid} {user_text}"

    # ── Start turn log ───────────────────────────────────────
    turn_log = logger.start_turn(
        session_id=session_id,
        user_text=user_text,
        has_image=img_input is not None,
        last_pid="",
        effective_query=effective_text,
        query_type=query_type,
    )
    raw_refs_html += f"**Turn:** `{turn_log.turn_id}`\n"
    raw_refs_html += f"**Effective query:** {effective_text}\n\n"

    try:
        # ════════════════════════════════════════════════════
        # PHASE 3: Retrieval (instrumented — hooks capture internals)
        # ════════════════════════════════════════════════════
        retrieval_result = retriever.search(
            query=effective_text,
            query_image=img_input if img_input else None,
            query_type=query_type,
        )

        # ── LOG step 1: Keyword ──────────────────────────────
        logger.log_keyword(turn_log, retriever._score_map_snapshot)

        # ── LOG step 2: Image ────────────────────────────────
        if img_input is not None:
            logger.log_image(
                turn_log,
                vit_hits_raw=retriever._vit_hits_raw,
                siglip_hits_raw=retriever._siglip_hits_raw,
                vit_series=retriever._vit_series or None,
                vit_dominant=retriever._vit_dominant,
                vit_top_score=retriever._vit_top_score,
                vit_confidence_threshold=_retriever_inner.cfg.vit_confidence_threshold,
            )

        # ── LOG step 3: Graph ────────────────────────────────
        logger.log_graph(
            turn_log,
            graph_context=retrieval_result.graph_context,
            triggered_by=retriever._graph_triggered_by,
        )

        # ── LOG step 4: Text ─────────────────────────────────
        logger.log_text(
            turn_log,
            hits_raw=retriever._text_hits_raw,
            skipped=retriever._vit_dominant,
            skip_reason="vit_dominant" if retriever._vit_dominant else "",
        )

        # ── LOG step 5: RRF ──────────────────────────────────
        logger.log_rrf(
            turn_log,
            score_map=retriever._score_map_snapshot,
            ranked=retriever._ranked_snapshot,
        )

        # Debug sidebar
        if retrieval_result.vit_series_filter:
            raw_refs_html += f"**ViT series:** {retrieval_result.vit_series_filter}\n"
        if retrieval_result.graph_context:
            raw_refs_html += f"**Graph:** {retrieval_result.graph_context.to_summary_str()}\n"
        raw_refs_html += "\n"

        if img_input and not retrieval_result.products and retrieval_result.image_search_used:
            ai_reply = (
                "❌ ระบบตรวจพบว่ารูปภาพที่อัปโหลดไม่เกี่ยวข้องกับสินค้าในระบบ "
                "กรุณาอัปโหลดภาพโซ่หรือสเตอร์ที่ชัดเจนครับ"
            )
            _log_finish(turn_log, ai_reply, [], t_start, error=None)
            chat_history += [
                {"role": "user",      "content": display_content, "_context": ""},
                {"role": "assistant", "content": ai_reply},
            ]
            return chat_history, chat_history, raw_refs_html, session_id

        # ════════════════════════════════════════════════════
        # PHASE 4: Rerank
        # inject ViT top hit เข้า query เมื่อ dominant=True
        # เพื่อให้ CE score ของรุ่นที่ ViT ชี้สูงกว่ารุ่นอื่นใน series เดียวกัน
        # ════════════════════════════════════════════════════
        rerank_query = user_text or "ค้นหาจากรูปภาพ"
        vit_hits     = retriever._vit_hits_raw or []
        vit_dominant = retriever._vit_dominant if hasattr(retriever, '_vit_dominant') else False

        if vit_dominant and vit_hits:
            # ดึง pid ของ ViT top hit (ScoredPoint — payload มี product_id)
            top_hit    = vit_hits[0]
            vit_top_pid = ""
            try:
                vit_top_pid = top_hit.payload.get("product_id", "") or top_hit.id
            except Exception:
                pass
            if vit_top_pid:
                rerank_query = f"{vit_top_pid} {rerank_query}".strip()

        reranked_result = reranker.rerank(
            query=rerank_query,
            retrieval_result=retrieval_result,
        )

        # ── LOG step 6: Reranker ─────────────────────────────
        logger.log_reranker(turn_log, reranked_result)

        llm_context: list[dict] = []
        for rank, p in enumerate(reranked_result.products, 1):
            pid = p.product_id
            raw_refs_html += (
                f"**{rank}. {pid}** "
                f"(final:{p.final_score:.3f} | CE:{p.cross_encoder_score:.2f} | "
                f"RRF:{p.rrf_score:.4f})\n"
            )
            if pid in product_db:
                enriched = _enrich_product(product_db[pid])
                enriched.update({
                    "_rank": rank,
                    "_final_score": round(p.final_score, 4),
                    "_best_chunk": p.best_chunk_text,
                })
                llm_context.append(enriched)

        if not llm_context:
            raw_refs_html += "⚠️ ไม่พบข้อมูลสินค้าที่ตรงกัน"

        # ════════════════════════════════════════════════════
        # PHASE 5: Gemini Agent
        # ════════════════════════════════════════════════════
        json_context       = json.dumps(llm_context, ensure_ascii=False, indent=2)
        graph_summary      = _build_graph_summary(retrieval_result.graph_context)
        system_instruction = _build_system_prompt(json_context, graph_summary)

        # Build gemini_history — inject _context จาก turn ก่อนเข้า user message
        # ทำให้ Gemini รู้ว่าแต่ละ turn มีข้อมูล product อะไร
        # แก้ปัญหา "สเตอร์นั้น" / "รุ่นนั้น" — Gemini เห็น context ครบทุก turn
        gemini_history = []
        for msg in chat_history:
            clean = re.sub(r'<[^>]+>', '', msg.get("content", "")).strip()
            if not clean:
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            ctx  = msg.get("_context", "")
            if ctx and role == "user":
                clean = f"[Product Context: {ctx}]\n{clean}"
            gemini_history.append({"role": role, "parts": [clean]})

        img_part = None
        if img_input:
            buf = BytesIO()
            img_input.convert("RGB").save(buf, format="JPEG")
            img_part = {"mime_type": "image/jpeg", "data": buf.getvalue()}

        model_chat = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_instruction,
            tools=[tracked_analyze, tracked_compare, tracked_filter],
        )
        chat_session = model_chat.start_chat(
            history=gemini_history,
            enable_automatic_function_calling=True,
        )
        prompt   = user_text.strip() or "ลูกค้าส่งรูปภาพมา ช่วยวิเคราะห์สเปกจาก RAG Context"
        response = chat_session.send_message([img_part, prompt] if img_part else [prompt])
        ai_reply = response.text

    except Exception as e:
        ai_reply  = f"❌ เกิดข้อผิดพลาด: {str(e)}"
        error_msg = str(e)
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()

    # ── LOG step 7+8: LLM + finish ───────────────────────────
    _log_finish(turn_log, ai_reply, _tool_call_log, t_start, error=error_msg)

    # เก็บ context_tag ติดไปกับ user turn
    # Gemini จะเห็น context นี้ใน turn ถัดไปผ่าน gemini_history
    context_tag = _build_context_tag(reranked_result, retrieval_result.graph_context if 'retrieval_result' in dir() else None)
    chat_history += [
        {"role": "user",      "content": display_content, "_context": context_tag},
        {"role": "assistant", "content": ai_reply},
    ]

    return chat_history, chat_history, raw_refs_html, session_id


def _log_finish(turn_log, ai_reply: str, tool_calls: list, t_start: float, error=None):
    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.log_llm(turn_log, ai_reply=ai_reply, tool_calls=tool_calls, latency_ms=latency_ms)
    logger.finish_turn(turn_log, error=error)
    print(
        f"📋 Logged {turn_log.turn_id}  "
        f"latency={latency_ms:.0f}ms  "
        f"top1={turn_log.summary_top1_pid}  "
        f"clarif={turn_log.summary_clarification}  "
        f"tools={turn_log.summary_tool_calls}"
    )


# ================================================================
# GRADIO UI
# ================================================================

css = ".gradio-container {max-width: 900px !important}"
js_code = """
function() {
    document.addEventListener('keydown', function(e) {
        const textbox = document.querySelector('#chat-textbox textarea');
        if (e.target === textbox && e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.querySelector('#submit-button').click();
        }
    });
}
"""

with gr.Blocks(css=css, js=js_code, title="Movex Sales AI Assistant v3") as demo:
    chat_state     = gr.State([])
    session_state  = gr.State("")

    gr.Markdown("<h1 style='text-align:center;'>🔗 Movex Sales AI Assistant v3</h1>")
    gr.Markdown(
        "<p style='text-align:center;color:#666;'>"
        "Jina v3 + ViT LoRA + SigLIP + GraphRAG + Cross-Encoder + Gemini"
        "</p>"
    )

    chatbot_ui = gr.Chatbot(height=600, type="messages", show_label=False)

    with gr.Row():
        img_input_ui = gr.Image(
            type="pil", show_label=False, container=False,
            scale=0, height=50, width=50,
            sources=["upload", "clipboard"], interactive=True,
        )
        txt_input = gr.Textbox(
            elem_id="chat-textbox", show_label=False,
            placeholder="พิมพ์ข้อความ หรือกดปุ่มรูปภาพด้านซ้าย...",
            scale=10, container=False, lines=1, max_lines=5,
        )
        submit_btn = gr.Button("➤", elem_id="submit-button", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button("🗑️ ล้างประวัติ", variant="secondary", size="sm")

    with gr.Accordion("🔍 ข้อมูลอ้างอิง + Log Info", open=False):
        with gr.Row():
            img_w = gr.Slider(0, 1, value=0.6, step=0.1, label="ความสำคัญรูปภาพ")
            txt_w = gr.Slider(0, 1, value=0.4, step=0.1, label="ความสำคัญข้อความ")
        raw_refs = gr.Markdown()
        gr.Markdown(
            f"📋 **Log files (rotate รายวัน):**\n"
            f"- `{logger.log_paths()['jsonl']}` — full structured log (JSONL)\n"
            f"- `{logger.log_paths()['csv']}` — summary spreadsheet (CSV)\n"
            f"- `{logger.log_paths()['txt']}` — human-readable debug (TXT)"
        )

    def reset_inputs():
        return None, ""

    submit_btn.click(
        fn=chat_interaction,
        inputs=[txt_input, img_input_ui, chat_state, img_w, txt_w, session_state],
        outputs=[chatbot_ui, chat_state, raw_refs, session_state],
    ).then(fn=reset_inputs, outputs=[img_input_ui, txt_input])

    txt_input.submit(
        fn=chat_interaction,
        inputs=[txt_input, img_input_ui, chat_state, img_w, txt_w, session_state],
        outputs=[chatbot_ui, chat_state, raw_refs, session_state],
    ).then(fn=reset_inputs, outputs=[img_input_ui, txt_input])

    clear_btn.click(
        fn=lambda: ([], [], "", logger.new_session()),
        inputs=None,
        outputs=[chatbot_ui, chat_state, raw_refs, session_state],
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)