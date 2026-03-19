"""
build_database.py
=================
Phase 3: สร้าง Qdrant Vector Database สำหรับ Movex Product Chatbot

Collections ที่สร้าง:
  1. movex_text_chunks  — Late Chunking ด้วย Jina v3 (1024 dim)
                          แต่ละ product แบ่งเป็นหลาย chunk (spec, usage, material)
                          ทุก chunk เห็น context ของทั้ง product ผ่าน late pooling

  2. movex_images       — Dual embedding ต่อรูป
                          - ViT LoRA vector  (768 dim) → specialist, แยก series ได้แม่น
                          - SigLIP vector    (768 dim) → generalist, text-image aligned

รัน:
  pip install qdrant-client transformers torch pillow tqdm peft
  pip install git+https://github.com/jina-ai/late-chunking.git  # หรือใช้ manual late chunking
  python build_database.py
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, OptimizersConfigDiff
)

# Models
from transformers import (
    AutoTokenizer, AutoModel,          # Jina v3
    ViTImageProcessor, ViTForImageClassification,  # ViT LoRA
    AutoProcessor,                     # SigLIP
)
from transformers import SiglipModel, SiglipProcessor
from peft import PeftModel

# ===============================================================
# CONFIG
# ===============================================================

@dataclass
class BuildConfig:
    # Paths
    metadata_path: str = "extracted_data_v3_machine_ready.json"
    image_base_dir: str = "image"
    qdrant_path: str = "./qdrant_db"          # local mode (ไม่ต้องรัน server)

    # Collections
    text_collection: str = "movex_text_chunks"
    image_collection: str = "movex_images"

    # Text embedding (Jina v3)
    jina_model_id: str = "jinaai/jina-embeddings-v3"
    text_dim: int = 1024
    max_tokens: int = 8192                    # Jina v3 รองรับ long context

    # Image embedding
    vit_base_model: str = "google/vit-base-patch16-224-in21k"
    vit_lora_path: str = "movex-lora-tuned-3"
    vit_num_labels: int = 10
    vit_dim: int = 768

    siglip_model_id: str = "google/siglip-base-patch16-224"
    siglip_dim: int = 768

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = BuildConfig()

# ===============================================================
# MODEL LOADERS
# ===============================================================

def load_jina_model(cfg: BuildConfig):
    """โหลด Jina v3 สำหรับ Late Chunking"""
    print("⏳ Loading Jina v3...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.jina_model_id, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        cfg.jina_model_id, trust_remote_code=True
    ).to(cfg.device)
    model.eval()
    print("✅ Jina v3 loaded")
    return tokenizer, model


def load_vit_lora(cfg: BuildConfig):
    """โหลด ViT + LoRA specialist model"""
    print("⏳ Loading ViT LoRA...")
    base = ViTForImageClassification.from_pretrained(
        cfg.vit_base_model, num_labels=cfg.vit_num_labels
    )
    model = PeftModel.from_pretrained(base, cfg.vit_lora_path)
    model.to(cfg.device)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(cfg.vit_base_model)
    print("✅ ViT LoRA loaded")
    return processor, model


def load_siglip(cfg: BuildConfig):
    """โหลด SigLIP generalist model"""
    print("⏳ Loading SigLIP...")
    processor = SiglipProcessor.from_pretrained(cfg.siglip_model_id)
    model = SiglipModel.from_pretrained(cfg.siglip_model_id).to(cfg.device)
    model.eval()
    print("✅ SigLIP loaded")
    return processor, model


# ===============================================================
# QDRANT SETUP
# ===============================================================

def setup_qdrant(cfg: BuildConfig) -> QdrantClient:
    """สร้าง Qdrant client และ collections"""
    client = QdrantClient(path=cfg.qdrant_path)

    # --- Text Collection ---
    # ใช้ named vectors: "jina" สำหรับ text embedding
    if client.collection_exists(cfg.text_collection):
        client.delete_collection(cfg.text_collection)

    client.create_collection(
        collection_name=cfg.text_collection,
        vectors_config={
            "jina": VectorParams(size=cfg.text_dim, distance=Distance.COSINE),
        },
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
    )
    print(f"✅ Collection '{cfg.text_collection}' created")

    # --- Image Collection ---
    # ใช้ named vectors 2 ชุด: "vit" และ "siglip"
    # ทำให้ search ด้วย vector ไหนก็ได้ หรือ fusion ก็ได้
    if client.collection_exists(cfg.image_collection):
        client.delete_collection(cfg.image_collection)

    client.create_collection(
        collection_name=cfg.image_collection,
        vectors_config={
            "vit":    VectorParams(size=cfg.vit_dim,    distance=Distance.COSINE),
            "siglip": VectorParams(size=cfg.siglip_dim, distance=Distance.COSINE),
        },
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
    )
    print(f"✅ Collection '{cfg.image_collection}' created")

    return client


# ===============================================================
# TEXT: LATE CHUNKING
# ===============================================================

def build_product_chunks(product: dict, material_db: dict, process_db: dict) -> list[dict]:
    """
    แบ่ง product ออกเป็น semantic chunks โดยรองรับ 2 product type:
      - Chain   : Plate_Width_mm, Weight_kg_m, Max_Working_Load_N,
                  Min_curve_radius_mm, temperature_range
      - Sprocket: Z_Teeth, Bore_mm, PD_mm, OD_mm, S_mm,
                  Sprocket_Type, compatible_chain

    Identity line ถูกฝังไว้ใน ทุก chunk เพื่อให้ Late Chunking
    มี context ของสินค้าแม้ตอน pool แยก chunk

    Returns list of {"chunk_type": str, "text": str}
    """
    pid = product.get("product_id", "")
    name = product.get("Ref") or product.get("Art_Nr") or pid
    product_type = product.get("product_type", "")
    is_sprocket = "sprocket" in product_type.lower() or "sprocket" in pid.lower()

    mat_code = product.get("Material", "")
    mat_info = material_db.get(mat_code, {})
    proc_code = product.get("manufacturing_process", "")
    proc_info = process_db.get(proc_code, {})

    chunks = []

    # ── Identity (ฝังไว้ทุก chunk) ──────────────────────────────
    # รวม compatible_chain และ Sprocket_Type ไว้ใน identity ถ้าเป็น sprocket
    # เพราะ user มักถามว่า "สเตอร์ที่ใช้กับโซ่ 820 ซีรีส์" ซึ่งต้องการข้อมูลนี้
    identity_parts = [
        f"Product: {name}",
        f"ID: {pid}",
        f"Series: {product.get('series', '-')}",
        f"Type: {product_type}",
    ]
    if is_sprocket:
        if product.get("Sprocket_Type"):
            identity_parts.append(f"Sprocket Type: {product['Sprocket_Type']}")
        if product.get("compatible_chain"):
            identity_parts.append(f"Compatible Chain: {product['compatible_chain']}")
    identity = " | ".join(identity_parts)

    # ── Chunk: Spec ─────────────────────────────────────────────
    specs_parts = []

    if is_sprocket:
        # Sprocket specs
        sprocket_fields = {
            "Z_Teeth":  "Teeth (Z)",
            "Bore_mm":  "Bore",
            "PD_mm":    "Pitch Diameter (PD)",
            "OD_mm":    "Outer Diameter (OD)",
            "S_mm":     "Hub Width (S)",
        }
        for key, label in sprocket_fields.items():
            val = product.get(key)
            if val is not None:
                unit = "mm" if key != "Z_Teeth" else ""
                specs_parts.append(f"{label}: {val}{' ' + unit if unit else ''}")

        # compatible chain ใน spec chunk ด้วย เพื่อเพิ่ม recall ตอน search
        if product.get("compatible_chain"):
            specs_parts.append(f"Compatible Chain Series: {product['compatible_chain']}")

    else:
        # Chain specs
        chain_fields = {
            "Plate_Width_mm":     "Plate Width",
            "Weight_kg_m":        "Weight",
            "Max_Working_Load_N": "Max Working Load",
            "Min_curve_radius_mm":"Min Curve Radius",
            "Pitch_mm":           "Pitch",
            "Radius_min_mm":      "Min Backflex Radius",
            "temperature_range":  "Temperature Range",
            "Links_m":            "Links per Meter",
        }
        for key, label in chain_fields.items():
            val = product.get(key)
            if val is not None:
                # เพิ่ม unit ให้ชัดเจน
                if key == "Plate_Width_mm":
                    specs_parts.append(f"{label}: {val} mm")
                elif key == "Weight_kg_m":
                    specs_parts.append(f"{label}: {val} kg/m")
                elif key == "Max_Working_Load_N":
                    specs_parts.append(f"{label}: {val} N")
                elif key in ("Min_curve_radius_mm", "Radius_min_mm", "Pitch_mm"):
                    specs_parts.append(f"{label}: {val} mm")
                else:
                    specs_parts.append(f"{label}: {val}")

    if specs_parts:
        chunks.append({
            "chunk_type": "spec",
            "text": f"{identity}\nSpecifications: {', '.join(specs_parts)}"
        })

    # ── Chunk: Usage & Application ──────────────────────────────
    usage_parts = []
    if product.get("usage"):
        usage_parts.append(f"Usage: {product['usage']}")
    if product.get("key_features"):
        usage_parts.append(f"Key Features: {product['key_features']}")
    if product.get("constraints"):
        usage_parts.append(f"Constraints: {product['constraints']}")

    if usage_parts:
        chunks.append({
            "chunk_type": "usage",
            "text": f"{identity}\n" + "\n".join(usage_parts)
        })

    # ── Chunk: Material & Manufacturing ─────────────────────────
    mat_parts = []
    if mat_info:
        mat_parts.append(
            f"Material: {mat_info.get('full_name', mat_code)} ({mat_code})"
        )
        if mat_info.get("description"):
            mat_parts.append(f"Material Properties: {mat_info['description']}")
    elif mat_code:
        mat_parts.append(f"Material: {mat_code}")

    if proc_info:
        mat_parts.append(
            f"Manufacturing Process: {proc_info.get('full_name', proc_code)} ({proc_code})"
        )
        if proc_info.get("description"):
            mat_parts.append(f"Process Notes: {proc_info['description']}")
    elif proc_code:
        mat_parts.append(f"Manufacturing Process: {proc_code}")

    if mat_parts:
        chunks.append({
            "chunk_type": "material",
            "text": f"{identity}\n" + "\n".join(mat_parts)
        })

    # ── Chunk: Compatible (sprocket เท่านั้น) ───────────────────
    # แยก chunk นี้ออกมาเพราะ user มักถามว่า "สเตอร์ตัวไหนใช้กับโซ่ X"
    # การมี dedicated chunk ช่วยให้ retrieve ตรงกว่าถ้า compatible_chain อยู่รวมกับ spec อื่น
    if is_sprocket and product.get("compatible_chain"):
        compatible_text = (
            f"{identity}\n"
            f"This sprocket is compatible with: {product['compatible_chain']}\n"
            f"Sprocket Type: {product.get('Sprocket_Type', '-')}\n"
            f"Manufacturing: {proc_info.get('full_name', proc_code) if proc_info else proc_code}"
        )
        chunks.append({
            "chunk_type": "compatible",
            "text": compatible_text
        })

    # ── Chunk: Full context (fallback) ──────────────────────────
    full_parts = [identity]
    if specs_parts:
        full_parts.append("Specifications: " + ", ".join(specs_parts))
    full_parts.extend(usage_parts)
    full_parts.extend(mat_parts)
    if is_sprocket and product.get("compatible_chain"):
        full_parts.append(f"Compatible Chain: {product['compatible_chain']}")

    chunks.append({
        "chunk_type": "full",
        "text": "\n".join(full_parts)
    })

    return chunks


@torch.no_grad()
def late_chunk_embed(
    chunks: list[dict],
    tokenizer,
    model,
    device: str,
    max_tokens: int = 8192,
) -> list[list[float]]:
    """
    Late Chunking: encode ทุก chunk รวมกันเป็น single long sequence
    แล้ว pool แยกตาม chunk boundary ทีหลัง

    ทำให้แต่ละ chunk embedding มี context ของทั้ง product ฝังอยู่
    ผ่าน self-attention ของ Transformer
    """
    texts = [c["text"] for c in chunks]

    # --- Tokenize แยกเพื่อรู้ boundary ของแต่ละ chunk ---
    chunk_encodings = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
        for t in texts
    ]

    # --- สร้าง combined input (concat tokens) ---
    # เพิ่ม [CLS] ที่ต้น และ [SEP] หลังแต่ละ chunk
    combined_ids = []
    boundaries = []  # (start, end) token index ของแต่ละ chunk
    cursor = 0

    for enc in chunk_encodings:
        ids = enc["input_ids"][0].tolist()
        # ตัด [CLS] และ [SEP] ออก (index 0 และ -1)
        ids_stripped = ids[1:-1]
        start = cursor
        combined_ids.extend(ids_stripped)
        cursor += len(ids_stripped)
        boundaries.append((start, cursor))

    # ตัดถ้ายาวเกิน
    max_len = min(max_tokens, tokenizer.model_max_length or 8192)
    combined_ids = combined_ids[:max_len]

    # เพิ่ม [CLS] นำหน้า
    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102
    full_ids = [cls_id] + combined_ids + [sep_id]

    input_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_tensor)

    # --- Forward pass ครั้งเดียว ---
    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    # ใช้ last hidden state
    hidden = outputs.last_hidden_state[0]  # (seq_len, dim)
    # offset +1 เพราะมี [CLS] นำหน้า
    offset = 1

    # --- Pool แต่ละ chunk แยกกัน (mean pooling) ---
    embeddings = []
    for start, end in boundaries:
        s = min(start + offset, hidden.shape[0] - 1)
        e = min(end + offset, hidden.shape[0])
        if s >= e:
            e = s + 1
        chunk_hidden = hidden[s:e]
        pooled = chunk_hidden.mean(dim=0)
        pooled = F.normalize(pooled, p=2, dim=0)
        embeddings.append(pooled.cpu().numpy().tolist())

    return embeddings


# ===============================================================
# IMAGE EMBEDDINGS
# ===============================================================

@torch.no_grad()
def embed_vit_lora(image: Image.Image, processor, model, device: str) -> list[float]:
    """ViT LoRA specialist — เก่ง detect series/รุ่น"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.base_model(
        pixel_values=inputs["pixel_values"],
        output_hidden_states=True,
        return_dict=True,
    )
    cls_emb = outputs.hidden_states[-1][:, 0, :]
    cls_emb = F.normalize(cls_emb, p=2, dim=1)
    return cls_emb.squeeze().cpu().numpy().tolist()


@torch.no_grad()
def embed_siglip(image: Image.Image, processor, model, device: str) -> list[float]:
    """SigLIP generalist — text-image aligned, รองรับ cross-modal search"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs)
    image_features = F.normalize(image_features, dim=-1)
    return image_features.squeeze().cpu().numpy().tolist()


# ===============================================================
# BUILD TEXT DATABASE
# ===============================================================

def build_text_db(
    client: QdrantClient,
    products: list[dict],
    material_db: dict,
    process_db: dict,
    jina_tokenizer,
    jina_model,
    cfg: BuildConfig,
):
    print("\n📝 Building text database (Late Chunking)...")
    points = []
    point_id = 0

    for product in tqdm(products, desc="Text chunks"):
        pid = product.get("product_id", "")
        chunks = build_product_chunks(product, material_db, process_db)

        if not chunks:
            continue

        try:
            embeddings = late_chunk_embed(
                chunks, jina_tokenizer, jina_model, cfg.device, cfg.max_tokens
            )
        except Exception as e:
            print(f"⚠️ Late chunking failed for {pid}: {e}")
            continue

        is_sprocket = "sprocket" in product.get("product_type", "").lower()

        for chunk, embedding in zip(chunks, embeddings):
            points.append(PointStruct(
                id=point_id,
                vector={"jina": embedding},
                payload={
                    "product_id": pid,
                    "chunk_type": chunk["chunk_type"],
                    "text": chunk["text"],
                    "series": product.get("series", ""),
                    "product_type": product.get("product_type", ""),
                    "material": product.get("Material", ""),
                    "is_sprocket": is_sprocket,

                    # ── Chain specs (None ถ้าไม่มี field นั้น) ──
                    "plate_width_mm":      product.get("Plate_Width_mm"),
                    "max_load_n":          product.get("Max_Working_Load_N"),
                    "weight_kg_m":         product.get("Weight_kg_m"),
                    "min_curve_radius_mm": product.get("Min_curve_radius_mm"),
                    "pitch_mm":            product.get("Pitch_mm"),

                    # ── Sprocket specs ───────────────────────────
                    "z_teeth":          product.get("Z_Teeth"),
                    "bore_mm":          product.get("Bore_mm"),
                    "pd_mm":            product.get("PD_mm"),
                    "od_mm":            product.get("OD_mm"),
                    "s_mm":             product.get("S_mm"),
                    "sprocket_type":    product.get("Sprocket_Type", ""),
                    "compatible_chain": product.get("compatible_chain", ""),
                }
            ))
            point_id += 1

    # Batch upsert
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=cfg.text_collection,
            points=points[i:i + batch_size],
        )

    print(f"✅ Text DB: {point_id} chunks จาก {len(products)} products")


# ===============================================================
# BUILD IMAGE DATABASE
# ===============================================================

def build_image_db(
    client: QdrantClient,
    products: list[dict],
    vit_processor, vit_model,
    siglip_processor, siglip_model,
    cfg: BuildConfig,
):
    print("\n🖼️ Building image database (ViT LoRA + SigLIP)...")
    points = []
    point_id = 0
    missing = 0

    for product in tqdm(products, desc="Images"):
        pid = product.get("product_id", "")

        for img_meta in product.get("images", []):
            img_path = os.path.join(
                cfg.image_base_dir,
                pid,
                os.path.basename(img_meta["image_path"])
            )

            # fallback: ลอง path ตรงๆ จาก JSON
            if not os.path.exists(img_path):
                img_path = img_meta["image_path"]

            if not os.path.exists(img_path):
                missing += 1
                continue

            try:
                image = Image.open(img_path).convert("RGB")

                vit_vec    = embed_vit_lora(image, vit_processor, vit_model, cfg.device)
                siglip_vec = embed_siglip(image, siglip_processor, siglip_model, cfg.device)

                points.append(PointStruct(
                    id=point_id,
                    vector={
                        "vit":    vit_vec,
                        "siglip": siglip_vec,
                    },
                    payload={
                        "product_id": pid,
                        "image_path": img_path,
                        "image_type": img_meta.get("type", "photo"),
                        "series": product.get("series", ""),
                        "product_type": product.get("product_type", ""),
                    }
                ))
                point_id += 1

            except Exception as e:
                print(f"⚠️ Error on {img_path}: {e}")

    # Batch upsert
    batch_size = 50
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=cfg.image_collection,
            points=points[i:i + batch_size],
        )

    print(f"✅ Image DB: {point_id} vectors | {missing} missing files")


# ===============================================================
# MAIN
# ===============================================================

def main():
    print("=" * 60)
    print("  Movex Database Builder v2.0")
    print(f"  Device: {CFG.device}")
    print("=" * 60)

    # Load JSON
    with open(CFG.metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    products    = data.get("products", [])
    material_db = {m["material_code"]: m for m in data.get("Materials", [])}
    process_db  = {p["manufacturing_process"]: p for p in data.get("manufacturing_process", [])}

    print(f"📦 Products: {len(products)} | Materials: {len(material_db)}")

    # Setup Qdrant
    client = setup_qdrant(CFG)

    # Load models
    jina_tokenizer, jina_model     = load_jina_model(CFG)
    vit_processor, vit_model       = load_vit_lora(CFG)
    siglip_processor, siglip_model = load_siglip(CFG)

    # Build databases
    build_text_db(
        client, products, material_db, process_db,
        jina_tokenizer, jina_model, CFG
    )

    build_image_db(
        client, products,
        vit_processor, vit_model,
        siglip_processor, siglip_model,
        CFG
    )

    # Summary
    text_count  = client.count(CFG.text_collection).count
    image_count = client.count(CFG.image_collection).count
    print("\n" + "=" * 60)
    print(f"  ✅ Build complete!")
    print(f"  Text chunks : {text_count}")
    print(f"  Image vectors: {image_count}")
    print(f"  DB path      : {CFG.qdrant_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
