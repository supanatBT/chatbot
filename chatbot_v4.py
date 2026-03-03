"""
chatbot_v4.py
=============
Movex Sales AI Assistant v4 — Adaptive Multi-Signal Pipeline

เปลี่ยนจาก v3:
  ✦ ใช้ RetrieverV4 — confidence-weighted RRF + multi-hop graph + series filter
  ✦ InstrumentedRetriever ใหม่ — hook เข้า v4 internal states โดยตรง
  ✦ Reranker ถูกรวมเข้า RetrieverV4.rank() แล้ว — ไม่มี CE model
  ✦ filter_product_specs ใช้ Qdrant payload filter ผ่าน retriever.filter_by_spec()
  ✦ _build_graph_summary รับ GraphResult แทน GraphContext
  ✦ ทุกอย่างอื่น (UI, logger, tools, system prompt) คงเดิม
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

from knowledge_graph import KnowledgeGraph
from retriever_v4 import RetrieverV4, RetrieverV4Config, ViTSignal, GraphResult, RankedProduct, RankedResult
from pipeline_logger import PipelineLogger

# ================================================================
# CONFIG
# ================================================================
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ ไม่พบ GEMINI_API_KEY")
QDRANT_PATH     = "./qdrant_db"
JSON_PATH       = "./extracted_data_v3_machine_ready.json"
KG_DB_PATH      = "./movex_kg.db"
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
print("  Movex Chatbot v4 — Adaptive Multi-Signal Pipeline")
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
    torch_dtype=torch.float32,
    attn_implementation="eager"
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

_retriever_inner = RetrieverV4(
    cfg=RetrieverV4Config(qdrant_path=QDRANT_PATH, device=DEVICE),
    jina_tokenizer=jina_tokenizer,
    jina_model=jina_model,
    vit_processor=vit_processor,
    vit_model=vit_model,
    siglip_processor=siglip_processor,
    siglip_model=siglip_model,
    product_db=product_db,
    kg=kg,
)
print("✅ RetrieverV4 + KnowledgeGraph")

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


def _build_graph_summary(graph_result) -> str:
    """
    สร้าง graph summary สำหรับ LLM
    รองรับทั้ง GraphResult (v4) และ GraphContext (v3 compat)
    """
    if graph_result is None:
        return "ไม่มี graph context"

    # v4 GraphResult
    if isinstance(graph_result, GraphResult):
        summary = graph_result.to_llm_context()
        return summary if summary else "ไม่มี graph context"

    # v3 GraphContext compat (fallback)
    parts = []
    if hasattr(graph_result, 'seed_products') and graph_result.seed_products:
        parts.append(f"[Seed products]: {', '.join(graph_result.seed_products)}")
    if hasattr(graph_result, 'compatible_products') and graph_result.compatible_products:
        parts.append(
            f"[Compatible parts สำหรับ {', '.join(graph_result.seed_products)}]: "
            f"{', '.join(graph_result.compatible_products[:5])}"
        )
    if hasattr(graph_result, 'same_series') and graph_result.same_series:
        parts.append(f"[สินค้า series เดียวกัน]: {', '.join(graph_result.same_series[:5])}")
    if hasattr(graph_result, 'material_details') and graph_result.material_details:
        for mat, info in graph_result.material_details.items():
            full = info.get("full_name", mat)
            desc = info.get("description", "")
            line = f"[วัสดุของ {', '.join(graph_result.seed_products)}]: รหัส={mat}, ชื่อ={full}"
            if desc:
                line += f", คุณสมบัติ={desc[:120]}"
            parts.append(line)
    if hasattr(graph_result, 'process_details') and graph_result.process_details:
        for proc, info in graph_result.process_details.items():
            full = info.get("full_name", proc)
            desc = info.get("description", "")
            adv  = info.get("advantages", "") or info.get("best_used_for", "")
            line = f"[กระบวนการผลิตของ {', '.join(graph_result.seed_products)}]: วิธี={full}"
            if desc:
                line += f", รายละเอียด={desc[:150]}"
            if adv:
                line += f", ข้อดี={adv[:100]}"
            parts.append(line)
    return "\n".join(parts) if parts else "ไม่มี graph context"


# ================================================================
# INSTRUMENTED RETRIEVER
# ================================================================

class InstrumentedRetriever:
    """
    Wrapper รอบ RetrieverV4 — expose internal states สำหรับ logger
    v4 ใช้ direct attribute แทน monkey-patch เพราะ v4 ออกแบบให้ inspect ได้
    """

    def __init__(self, inner: RetrieverV4):
        self.inner = inner
        self._reset()

    def _reset(self):
        self._vit_signal:        ViTSignal = ViTSignal()
        self._graph_result:      GraphResult = GraphResult()
        self._score_map_snapshot: dict = {}
        self._ranked_snapshot:   list = []
        self._graph_triggered_by: str = "none"

        # compat fields สำหรับ logger v3
        self._vit_hits_raw:     list = []
        self._siglip_hits_raw:  list = []
        self._vit_top_score:    float = 0.0
        self._vit_series:       str = ""
        self._vit_dominant:     bool = False
        self._text_hits_raw:    list = []

    def search(self, query, query_image=None, query_type="general"):
        self._reset()
        result = self.inner.search(
            query=query,
            query_image=query_image,
            query_type=query_type,
        )

        # capture v4 states
        if result.vit_signal:
            sig = result.vit_signal
            self._vit_signal     = sig
            self._vit_top_score  = sig.confidence
            self._vit_series     = sig.series or ""
            self._vit_dominant   = sig.dominant

        if result.graph_result:
            gr = result.graph_result
            self._graph_result = gr
            if gr.seed_products:
                self._graph_triggered_by = (
                    "vit" if result.vit_signal and result.vit_signal.series
                    else "keyword"
                )

        # capture score_map จาก result โดยตรง (แก้ bug seeds=[])
        self._score_map_snapshot = getattr(result, '_score_map', {})
        self._ranked_snapshot    = getattr(result, '_ranked', [])

        return result

    @property
    def cfg(self):
        return self.inner.cfg


retriever = InstrumentedRetriever(_retriever_inner)


# ================================================================
# TOOL CALL TRACKER
# ================================================================

_tool_call_log: list[dict] = []

def _tracked(fn):
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

def analyze_product_database(
    category: str = "all",
    series: str = "",
) -> str:
    """
    วิเคราะห์ฐานข้อมูลสินค้า Movex ทั้งหมด
    ใช้เมื่อลูกค้าถามภาพรวม เช่น "มีสินค้ากี่รุ่น" "มีอะไรบ้าง"

    Parameters
    ----------
    category : "all" | "chain" | "sprocket"
    series   : กรอง series เช่น "820" "880" (ว่างหมายถึงทั้งหมด)
    """
    filtered = {}
    for pid, p in product_db.items():
        if category == "chain" and "chain" not in pid.lower():
            continue
        if category == "sprocket" and "sprocket" not in pid.lower():
            continue
        if series and series not in str(p.get("series", "")):
            continue
        filtered[pid] = p

    if not filtered:
        return f"ไม่พบสินค้าในหมวด {category} series {series}"

    chains    = [pid for pid in filtered if "chain"    in pid.lower()]
    sprockets = [pid for pid in filtered if "sprocket" in pid.lower()]

    result = f"พบสินค้าทั้งหมด {len(filtered)} รุ่น\n"
    result += f"  Chain: {len(chains)} รุ่น\n"
    result += f"  Sprocket: {len(sprockets)} รุ่น\n"
    if chains:
        result += f"  Chain list: {', '.join(sorted(chains)[:15])}\n"
    if sprockets:
        result += f"  Sprocket list: {', '.join(sorted(sprockets)[:10])}\n"
    return result


def compare_specific_products(product_ids: str) -> str:
    """
    เปรียบเทียบสินค้าหลายรุ่นแบบ side-by-side
    ใช้เมื่อลูกค้าต้องการเปรียบเทียบ 2 รุ่นขึ้นไป

    Parameters
    ----------
    product_ids : รหัสสินค้าคั่นด้วยจุลภาค เช่น "LF820_K325,LF820_K750"
    """
    ids = [x.strip() for x in product_ids.split(",")]
    results = []
    for pid in ids:
        # ค้นหาแบบ fuzzy ถ้าไม่เจอ exact match
        p = product_db.get(pid)
        if not p:
            for k in product_db:
                if pid.lower() in k.lower() or k.lower() in pid.lower():
                    p = product_db[k]
                    pid = k
                    break
        if not p:
            results.append(f"{pid}: ไม่พบในระบบ")
            continue
        specs = {
            "series":         p.get("series", "-"),
            "product_type":   p.get("product_type", "-"),
            "plate_width_mm": p.get("plate_width_mm", "-"),
            "pitch_mm":       p.get("pitch_mm", "-"),
            "max_load_n":     p.get("Max_Working_Load_N", p.get("max_load_n", "-")),
            "material":       p.get("Material", "-"),
            "min_curve_radius_mm": p.get("Min_curve_radius_mm", p.get("Radius_min_mm", "-")),
            "Ref":            p.get("Ref", "-"),
        }
        lines = [f"=== {pid} ==="]
        for k, v in specs.items():
            lines.append(f"  {k}: {v}")
        results.append("\n".join(lines))

    return "\n\n".join(results) if results else "ไม่พบสินค้าที่ระบุ"


def filter_product_specs(
    spec_type: str,
    condition: str,
    target_value: float,
    product_type: str = None,
    sort_by: str = None,    
    sort_order: str = "asc",     
) -> str:
    """
    กรองสินค้าตาม spec — ใช้ Qdrant payload filter (v4)
    ใช้เมื่อลูกค้าระบุเงื่อนไขตัวเลข เช่น "กว้างไม่เกิน 100mm" "รับโหลดมากกว่า 5000N"

    Parameters
    ----------
    spec_type    : "width" | "load" | "radius" | "pitch" | "weight" | "teeth" | "bore"
    condition    : "<" | "<=" | ">" | ">=" | "=="
    target_value : ค่าตัวเลข เช่น 100.0
    """
    # ลอง Qdrant filter ก่อน — เร็วกว่า JSON loop มาก
    try:
        hits = _retriever_inner.filter_by_spec(spec_type, condition, target_value)
        if hits:
            lines = [f"พบ {len(hits)} รุ่นที่ {spec_type} {condition} {target_value}:"]
            for pid, val in sorted(hits, key=lambda x: str(x[0])):
                lines.append(f"  {pid} ({val})")
            return "\n".join(lines)
    except Exception as e:
        print(f"⚠️ Qdrant filter fallback to JSON: {e}")

    # Fallback: JSON loop (compat กับ field ที่ยังไม่ได้ index ใน Qdrant)
    key_map = {
        "width":   ["plate_width_mm", "Width_mm"],
        "load":    ["Max_Working_Load_N", "max_load_n"],
        "radius":  ["Min_curve_radius_mm", "Radius_min_mm"],
        "pitch":   ["pitch_mm", "Pitch_mm"],
        "weight":  ["weight_kg_m"],
        "teeth":   ["z_teeth"],
        "bore":    ["bore_mm"],
    }
    target_keys = key_map.get(spec_type.lower(), [spec_type])
    matched = []
    for pid, p in product_db.items():
        val = None
        for key in target_keys:
            if key in p and p[key] is not None:
                try:
                    val = float(p[key])
                except (ValueError, TypeError):
                    nums = re.findall(r'\d+\.?\d*', str(p[key]))
                    val = float(nums[0]) if nums else None
                if val is not None:
                    break
        if val is None and spec_type.lower() == "width":
            km = re.search(r'K(\d{3,4})', pid)
            if km:
                val = (float(km.group(1)) / 100) * 25.4
        if val is not None:
            if product_type and product_type.lower() not in p.get("product_type", "").lower():
                continue
            ok = (condition == "<"  and val <  target_value or
                  condition == "<=" and val <= target_value or
                  condition == ">"  and val >  target_value or
                  condition == ">=" and val >= target_value or
                  condition == "==" and val == target_value)
            if ok:
                matched.append(f"{pid} ({val})")
    if sort_by and matched:
        matched.sort(key=lambda x: float(re.search(r'[\d.]+', x.split('(')[1]).group()), 
                     reverse=(sort_order == "desc"))            
    if not matched:
        return f"ไม่พบสินค้าที่มี {spec_type} {condition} {target_value}"
    return f"พบ {len(matched)} รุ่น ได้แก่: {', '.join(matched)}"


tracked_analyze = _tracked(analyze_product_database)
tracked_compare = _tracked(compare_specific_products)
tracked_filter  = _tracked(filter_product_specs)


# ================================================================
# SYSTEM PROMPT
# ================================================================

def _build_system_prompt(json_context: str, graph_summary: str,vit_signal=None) -> str:
    vit_context = ""
    if vit_signal and vit_signal.series:
        vit_context = f"""
    [ViT Signal — ผลการวิเคราะห์รูปภาพ]
    series     : {vit_signal.series}
    confidence : {vit_signal.confidence:.3f}
    dominant   : {"✅ มั่นใจสูง (≥0.85)" if vit_signal.dominant else "⚠️ ปานกลาง (0.50–0.85)"}
    top_pid    : {vit_signal.top_pid or "-"}
    """
    return f"""
คุณคือ 'Movex Assistant' วิศวกรฝ่ายขายมืออาชีพของบริษัท Movex
ตอบภาษาไทย เป็นธรรมชาติ มีหางเสียง (ครับ/ค่ะ)

{vit_context}
[RAG Context — สินค้าที่ระบบค้นหามาได้ เรียงตาม final_score สูงสุด]
{json_context}

[GraphRAG Context — ความสัมพันธ์ของสินค้า]
{graph_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 ขั้นตอนที่ 1 (บังคับ): ตรวจก่อนตอบทุกครั้ง
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ก่อนตอบทุกครั้ง ให้ถามตัวเองว่า "ลูกค้าระบุรุ่นสินค้าชัดเจนแล้วหรือยัง?"

✅ ระบุชัด → ตอบ spec ได้เลย ถ้ามีอย่างน้อย 1 ข้อต่อไปนี้:
  • รหัสรุ่น / Ref ชัดเจน เช่น "LF820 K325", "54901"
  • series + ความกว้าง เช่น "880 กว้าง 82.5"
  • รูปภาพที่ ViT confidence ≥ 0.90 และ dominant=✅
    → ดู product_type ของ rank 1 ใน RAG Context แล้วแยกการตอบ:

    ถ้า product_type = "Chain":
      → ตอบ spec ของ rank 1 ได้เลย
      → ท้ายคำตอบเสนอ width อื่นใน series เดียวกัน
         พร้อมบอกว่าแต่ละ width เหมาะกับสินค้าประเภทใด

    ถ้า product_type = "Sprocket" หรือ "Sideflexing Sprocket" หรือ "Heavy-Duty Sprocket":
      → ตอบ spec ของ rank 1 ได้เลย (Z-teeth, Bore, PD, OD, compatible chain)
      → ไม่ต้องถาม Z-teeth หรือ Bore เพิ่ม เพราะ retrieval ระบุรุ่นชัดแล้ว
      → ท้ายคำตอบบอก compatible chain จาก GraphRAG Context

⚠️ รู้ series แต่ยังไม่รู้ width = ถามแบบนี้:
  • รูปภาพ Chain ที่ ViT confidence ≥ 0.50 แต่ < 0.90
    → บอก series ที่ detect ได้ก่อน
    → ถามเฉพาะ width + การใช้งาน
    → เสนอตัวเลือก width ทั้งหมดใน series นั้น
       พร้อมบอกว่าแต่ละ width เหมาะกับสินค้าประเภทใด
    ตัวอย่าง:
      "จากรูปเป็น 880 Series Sideflexing ครับ มี width ให้เลือก:
       • 82.5 mm — ขวด PET / กระป๋อง single file
       • 114.3 mm — ขวดแก้ว / กล่องที่ต้องการความเสถียรมากขึ้น
       ต้องการ width ไหนครับ?"

❌ ยังไม่ระบุ → ถามกลับก่อนเสมอ ห้ามดู spec ห้ามตอบรุ่นใดรุ่นหนึ่ง:
  • ไม่มี series ไม่มี width เช่น "มีสายพานเลี้ยวได้ไหม", "แนะนำสายพานหน่อย"
  • รูปภาพที่ ViT confidence < 0.50
  • รูปภาพที่ top-1 กับ top-2 rank gap < 0.003

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
- ยึดข้อมูลจาก rank 1 เป็นหลัก
- ห้ามมั่วตัวเลขหรือรหัสสินค้าเด็ดขาด
- "เลี้ยวได้ไหม" → ดู product_type ก่อนเสมอ (ห้ามดู radius อย่างเดียว)
  product_type = "Straight Running Chain"  → เลี้ยวไม่ได้ ตอบทันทีโดยไม่ต้องดู radius
  product_type = "Sideflexing Chain"       → เลี้ยวได้ แล้วค่อยบอก Min_curve_radius_mm
- "workload" → ดู Max_Working_Load_N
- "ใช้สเตอร์ตัวไหน" → ตอบแค่รหัสสเตอร์ที่ compatible จาก GraphRAG Context
  ห้ามอธิบาย spec ยาว ตอบแค่ชื่อรุ่น + จำนวนฟัน (ถ้ามี) เท่านั้น
- "สเตอร์นั้นผลิตยังไง" → ถ้า GraphRAG มีสเตอร์ compatible หลายตัว ต้องตอบให้ครบทุกตัว
  ห้ามตอบแค่ตัวเดียวถ้า context มีมากกว่า 1
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
5. query มีคำต่อไปนี้ → เรียก filter_product_specs ทันทีก่อนตอบ ห้ามตอบจาก RAG Context โดยตรง:
   - "เบาสุด / น้ำหนักน้อยสุด / lightweight"  → spec_type="weight", condition=">=", target_value=0, sort_by="weight", sort_order="asc"
   - "รับโหลดสูงสุด / แข็งแรงสุด"             → spec_type="load",   condition=">=", target_value=0, sort_by="load",   sort_order="desc"
   - "กว้างสุด / กว้างมากสุด"                  → spec_type="width",  condition=">=", target_value=0, sort_by="width",  sort_order="desc"
   - "width ≥ X / กว้างอย่างน้อย X mm"         → spec_type="width",  condition=">=", target_value=X, product_type="chain"
"""


# ================================================================
# MAIN CHAT FUNCTION
# ================================================================

def chat_interaction(
    user_text: str,
    img_input,
    chat_history: list,
    last_pids: list = None,
    session_id: str = "",
    last_vit_signal = None,
    last_graph_result = None,
):
    global _tool_call_log

    if last_pids is None:
        last_pids = []

    if not user_text.strip() and img_input is None:
        return (chat_history, chat_history, "⚠️ กรุณาพิมพ์ข้อความหรืออัปโหลดรูปภาพ",
                last_pids, session_id, last_vit_signal, last_graph_result)

    if not session_id:
        session_id = logger.new_session()

    _tool_call_log = []
    t_start = time.perf_counter()

    # ── Display content ────────────────────────────────────
    display_content = ""
    if img_input:
        display_content += image_to_base64_html(img_input) + "<br>"
    if user_text:
        display_content += user_text
    if not display_content:
        display_content = "[แนบรูปภาพ]"

    raw_refs_html = "### 🔍 ข้อมูลอ้างอิง (Debug v4)\n\n"
    ai_reply      = ""
    error_msg     = None

    # ── Detect follow-up ────────────────────────────────────
    # follow-up = ไม่มีรูปใหม่ + พิมพ์สั้น + มี context จาก turn ก่อน
    is_followup = (
        img_input is None and
        user_text.strip() and
        len(user_text.split()) <= 6 and
        bool(last_pids or last_vit_signal or last_graph_result)
    )

    # ── Effective text — carry-over last_pids ───────────────
    effective_text = user_text or "ลูกค้าส่งรูปภาพมา"
    if is_followup and last_pids:
        pid_prefix = " ".join(last_pids[:3])
        effective_text = f"{pid_prefix} {user_text}"

    query_type = "image" if img_input is not None else "general"

    raw_refs_html += f"**follow-up:** {is_followup} | **last_pids:** {last_pids}\n"

    # ── Start turn log ──────────────────────────────────────
    turn_log = logger.start_turn(
        session_id=session_id,
        user_text=user_text,
        has_image=img_input is not None,
        last_pid=" ".join(last_pids) if last_pids else "",
        effective_query=effective_text,
        query_type=query_type,
    )
    raw_refs_html += f"**Turn:** `{turn_log.turn_id}`\n"
    raw_refs_html += f"**Effective query:** {effective_text}\n\n"

    try:
        # ════════════════════════════════════════════════════
        # PHASE 3: Retrieval
        # carry-over vit_signal จาก turn ก่อนถ้าเป็น follow-up ไม่มีรูปใหม่
        # ════════════════════════════════════════════════════
        retrieval_result = retriever.search(
            query=effective_text,
            query_image=img_input if img_input else None,
            query_type=query_type,
        )

        # ── carry-over vit_signal เมื่อไม่มีรูปใหม่ ─────────
        vit_sig = retriever._vit_signal
        if is_followup and not vit_sig.series and last_vit_signal:
            # ViT ไม่ได้รันเพราะไม่มีรูป ใช้ signal จาก turn ก่อนแทน
            vit_sig = last_vit_signal
            raw_refs_html += f"**ViT (carry-over):** series={vit_sig.series} conf={vit_sig.confidence:.3f}\n"

        # ── carry-over graph_result เมื่อเป็น follow-up ─────
        if is_followup and last_graph_result:
            # inject graph ของ turn ก่อนเข้า retrieval result
            # เพื่อให้ LLM เห็น compatible/process ของ turn ก่อนโดยไม่ต้อง traverse ใหม่
            if not (retrieval_result.graph_result and
                    retrieval_result.graph_result.seed_products):
                retrieval_result.graph_result = last_graph_result
                retrieval_result.graph_context = retriever.inner._graph_result_to_context(last_graph_result)
                raw_refs_html += f"**Graph (carry-over):** seeds={last_graph_result.seed_products}\n"
        if vit_sig.series:
            raw_refs_html += (
                f"**ViT series:** {vit_sig.series} "
                f"(conf={vit_sig.confidence:.3f}, "
                f"dominant={'✅' if vit_sig.dominant else '❌'})\n"
            )
        if retrieval_result.graph_result and retrieval_result.graph_result.seed_products:
            gr = retrieval_result.graph_result
            raw_refs_html += (
                f"**Graph:** seeds={gr.seed_products} | "
                f"compatible={gr.compatible[:3]} | "
                f"hop_trace={gr.hop_trace}\n"
            )
        raw_refs_html += "\n"

        # ── LOG ──────────────────────────────────────────────
        logger.log_keyword(turn_log, retriever._score_map_snapshot)

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

        logger.log_graph(
            turn_log,
            graph_context=retrieval_result.graph_context,
            triggered_by=retriever._graph_triggered_by,
        )

        logger.log_text(
            turn_log,
            hits_raw=retriever._text_hits_raw,
            skipped=False,   # v4 ไม่ skip text search แม้ ViT dominant
            skip_reason="",
        )

        logger.log_rrf(
            turn_log,
            score_map=retriever._score_map_snapshot,
            ranked=retriever._ranked_snapshot,
        )

        # ── Image not found ─────────────────────────────────
        if img_input and not retrieval_result.products and retrieval_result.image_search_used:
            ai_reply = (
                "❌ ระบบตรวจพบว่ารูปภาพที่อัปโหลดไม่เกี่ยวข้องกับสินค้าในระบบ "
                "กรุณาอัปโหลดภาพโซ่หรือสเตอร์ที่ชัดเจนครับ"
            )
            _log_finish(turn_log, ai_reply, [], t_start, error=None)
            chat_history += [
                {"role": "user",      "content": display_content},
                {"role": "assistant", "content": ai_reply},
            ]
            return chat_history, chat_history, raw_refs_html, last_pid, session_id

        # ════════════════════════════════════════════════════
        # PHASE 4: Rank — Keyword Boost + Final Sort
        # (ย้ายมาจาก reranker.py — ไม่มี CE model แล้ว)
        # ════════════════════════════════════════════════════
        retrieval_result.vit_signal = vit_sig
        ranked_result = _retriever_inner.rank(retrieval_result)
        raw_refs_html += f"**Mode:** RRF+KeywordBoost (no CE)\n"

        logger.log_reranker(turn_log, ranked_result)

        # ── Build LLM context ───────────────────────────────
        llm_context: list[dict] = []
        for rank, p in enumerate(ranked_result.products, 1):
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
        json_context  = json.dumps(llm_context, ensure_ascii=False, indent=2)
        graph_summary = _build_graph_summary(retrieval_result.graph_result)
        system_instruction = _build_system_prompt(json_context, graph_summary, vit_signal=vit_sig)

        gemini_history = []
        for msg in chat_history:
            clean = re.sub(r'<[^>]+>', '', msg.get("content", "")).strip()
            if clean:
                role = "model" if msg["role"] == "assistant" else "user"
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

    _log_finish(turn_log, ai_reply, _tool_call_log, t_start, error=error_msg)

    chat_history += [
        {"role": "user",      "content": display_content},
        {"role": "assistant", "content": ai_reply},
    ]

    # ── Extract pids จาก reply + reranked products ──────────
    # collect จาก 2 แหล่ง: ai_reply + reranked top products
    new_last_pids = []
    try:
        # 1. จาก reranked products — ครบที่สุด
        if "ranked_result" in dir() and ranked_result:
            for p in ranked_result.products[:3]:
                if p.product_id not in new_last_pids:
                    new_last_pids.append(p.product_id)

        # 2. จาก ai_reply — pid ที่ LLM พูดถึงจริงๆ (เพิ่มเข้าถ้ายังไม่มี)
        for m in re.finditer(
            r'\b(LF\s*\d+\s*(?:TAB\s*)?[A-Z]?\d+|LFN\s*\d+\s*\w+|\d{5})\b',
            ai_reply, re.IGNORECASE
        ):
            pid = m.group().strip()
            if pid not in new_last_pids:
                new_last_pids.append(pid)
            if len(new_last_pids) >= 5:
                break
    except Exception:
        pass

    # ── carry-over vit_signal และ graph_result ──────────────
    new_vit_signal    = vit_sig if vit_sig and vit_sig.series else last_vit_signal
    new_graph_result  = None
    try:
        if retrieval_result.graph_result and retrieval_result.graph_result.seed_products:
            new_graph_result = retrieval_result.graph_result
        else:
            new_graph_result = last_graph_result
    except Exception:
        new_graph_result = last_graph_result

    return (chat_history, chat_history, raw_refs_html,
            new_last_pids, session_id, new_vit_signal, new_graph_result)


def _log_finish(turn_log, ai_reply: str, tool_calls: list, t_start: float, error=None):
    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.log_llm(turn_log, ai_reply=ai_reply, tool_calls=tool_calls, latency_ms=latency_ms)
    logger.finish_turn(turn_log, error=error)
    print(
        f"📋 Logged {turn_log.turn_id}  "
        f"latency={latency_ms:.0f}ms  "
        f"top1={turn_log.summary_top1_pid}  "
        f"vit_conf={retriever._vit_top_score:.3f}  "
        f"tools={turn_log.summary_tool_calls}"
    )


# ================================================================
# GRADIO UI — คงเดิมทุกอย่าง
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

with gr.Blocks(css=css, js=js_code, title="Movex Sales AI Assistant v4") as demo:
    chat_state          = gr.State([])
    last_pids_state     = gr.State([])        # list of pids จาก turn ก่อน
    session_state       = gr.State("")
    last_vit_state      = gr.State(None)      # ViTSignal carry-over
    last_graph_state    = gr.State(None)      # GraphResult carry-over

    gr.Markdown("<h1 style='text-align:center;'>🔗 Movex Sales AI Assistant v4</h1>")
    gr.Markdown(
        "<p style='text-align:center;color:#666;'>"
        "Jina v3 + ViT LoRA + SigLIP + Multi-hop GraphRAG + "
        "Confidence-Weighted RRF + Enriched Reranker + Gemini"
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
        raw_refs = gr.Markdown()
        gr.Markdown(
            f"📋 **Log files (rotate รายวัน):**\n"
            f"- `{logger.log_paths()['jsonl']}` — full structured log (JSONL)\n"
            f"- `{logger.log_paths()['csv']}` — summary spreadsheet (CSV)\n"
            f"- `{logger.log_paths()['txt']}` — human-readable debug (TXT)"
        )

    def reset_inputs():
        return None, ""

    _inputs  = [txt_input, img_input_ui, chat_state,
                last_pids_state, session_state, last_vit_state, last_graph_state]
    _outputs = [chatbot_ui, chat_state, raw_refs,
                last_pids_state, session_state, last_vit_state, last_graph_state]

    submit_btn.click(
        fn=chat_interaction,
        inputs=_inputs,
        outputs=_outputs,
    ).then(fn=reset_inputs, outputs=[img_input_ui, txt_input])

    txt_input.submit(
        fn=chat_interaction,
        inputs=_inputs,
        outputs=_outputs,
    ).then(fn=reset_inputs, outputs=[img_input_ui, txt_input])

    clear_btn.click(
        fn=lambda: ([], [], "", [], "", None, None, logger.new_session()),
        inputs=None,
        outputs=[chatbot_ui, chat_state, raw_refs,
                 last_pids_state, session_state, last_vit_state, last_graph_state],
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)