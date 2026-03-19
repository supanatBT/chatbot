"""
analyze_vit_embedding.py
========================
เปรียบเทียบ embedding จาก:
  A) model.base_model() — pretrained ViT bypass LoRA (v1 style)
  B) model.base_model.model.vit() — LoRA-adapted ViT encoder

วิธีรัน:
  pip install torch transformers peft pillow scikit-learn matplotlib seaborn
  python analyze_vit_embedding.py
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from peft import PeftModel
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ===============================
# CONFIG
# ===============================
METADATA_PATH   = "extracted_data_v3_machine_ready.json"
LORA_PATH       = "movex-lora-tuned-3"
BASE_MODEL_ID   = "google/vit-base-patch16-224-in21k"
IMAGE_BASE_DIR  = "image"
NUM_LABELS      = 10
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")

processor = ViTImageProcessor.from_pretrained(BASE_MODEL_ID)

base_vit = ViTForImageClassification.from_pretrained(
    BASE_MODEL_ID, num_labels=NUM_LABELS
)
lora_model = PeftModel.from_pretrained(base_vit, LORA_PATH)
lora_model.to(DEVICE)
lora_model.eval()

print("Model loaded")

# debug: ดู structure
print("\nModel structure:")
print(f"  type(lora_model)            : {type(lora_model)}")
print(f"  type(lora_model.base_model) : {type(lora_model.base_model)}")

# ===============================
# EMBEDDING FUNCTIONS
# ===============================

@torch.no_grad()
def embed_bypass_lora(image: Image.Image) -> np.ndarray:
    """v1 style — เรียก base_model โดยตรง bypass LoRA layers"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = lora_model.base_model(
        pixel_values=inputs["pixel_values"],
        output_hidden_states=True,
        return_dict=True,
    )
    cls = outputs.hidden_states[-1][:, 0, :]
    cls = F.normalize(cls, p=2, dim=1)
    return cls.squeeze().cpu().numpy()


@torch.no_grad()
def embed_with_lora(image: Image.Image) -> np.ndarray:
    """LoRA-adapted — เรียก lora_model โดยตรง LoRA layers ทำงาน"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    # เรียก lora_model ตรงๆ — PeftModel forward ผ่าน LoRA adapters
    outputs = lora_model(
        pixel_values=inputs["pixel_values"],
        output_hidden_states=True,
        return_dict=True,
    )
    cls = outputs.hidden_states[-1][:, 0, :]
    cls = F.normalize(cls, p=2, dim=1)
    return cls.squeeze().cpu().numpy()


@torch.no_grad()
def embed_logits(image: Image.Image) -> np.ndarray:
    """ใช้ logits (10-dim) เป็น embedding — บอก classification confidence ตรงๆ"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = lora_model(pixel_values=inputs["pixel_values"])
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs.squeeze().cpu().numpy()


@torch.no_grad()
def get_classification(image: Image.Image):
    """ดู series prediction จาก logits"""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = lora_model(pixel_values=inputs["pixel_values"])
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()
    conf = probs[0, pred].item()
    label = lora_model.config.id2label.get(pred, str(pred))
    return label, conf, probs.squeeze().cpu().numpy()


# ===============================
# LOAD IMAGES
# ===============================
print("\nLoading images...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images_data = []

for p in data.get("products", []):
    pid = p.get("product_id", "")
    series = str(p.get("series", "unknown"))
    for img_meta in p.get("images", []):
        img_path = os.path.join(
            IMAGE_BASE_DIR, pid,
            os.path.basename(img_meta["image_path"])
        )
        if not os.path.exists(img_path):
            img_path = img_meta.get("image_path", "")
        if os.path.exists(img_path):
            images_data.append((pid, series, img_path))

print(f"Found {len(images_data)} images from {len(set(d[0] for d in images_data))} products")

# ===============================
# EXTRACT EMBEDDINGS
# ===============================
print("Extracting embeddings (3 variants)...")

bypass_vecs = []
lora_vecs   = []
logit_vecs  = []
labels      = []
series_labels = []
pids        = []
classifications = []

for pid, series, img_path in images_data:
    try:
        img = Image.open(img_path)
        bypass_vecs.append(embed_bypass_lora(img))
        lora_vecs.append(embed_with_lora(img))
        logit_vecs.append(embed_logits(img))
        pred_label, conf, probs = get_classification(img)
        classifications.append((pred_label, conf))
        labels.append(pid.replace("movex_chain_", "").replace("movex_sprocket_", ""))
        series_labels.append(series)
        pids.append(pid)
    except Exception as e:
        print(f"  Error {img_path}: {e}")

bypass_vecs = np.array(bypass_vecs)
lora_vecs   = np.array(lora_vecs)
logit_vecs  = np.array(logit_vecs)

print(f"Extracted {len(bypass_vecs)} embeddings")
print(f"  bypass dim : {bypass_vecs.shape[1]}")
print(f"  lora dim   : {lora_vecs.shape[1]}")
print(f"  logits dim : {logit_vecs.shape[1]}")

# ===============================
# ANALYSIS 1: ตรวจว่า bypass กับ lora ต่างกันไหม
# ===============================
print("\n=== ตรวจสอบ bypass vs lora ===")
diff = np.abs(bypass_vecs - lora_vecs).mean()
print(f"Mean absolute diff (bypass vs lora hidden state): {diff:.8f}")
if diff < 1e-6:
    print("  ⚠️  เหมือนกันทุกตัว — LoRA ไม่ affect hidden_states")
    print("  → LoRA อาจ inject เฉพาะ attention weights แต่ hidden_state output ไม่ต่าง")
    print("  → ใช้ logits แทน hidden state สำหรับ LoRA-specific embedding")
else:
    print(f"  ✅ ต่างกัน — LoRA ทำงานและ affect hidden states")

# ===============================
# ANALYSIS 2: Intra/Inter class similarity
# ===============================
print("\n=== Intra vs Inter class cosine similarity ===")

def compute_class_distances(vecs, labels):
    sim_matrix = cosine_similarity(vecs)
    intra, inter = [], []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                intra.append(sim_matrix[i, j])
            else:
                inter.append(sim_matrix[i, j])
    return np.array(intra), np.array(inter)

results = {}
for name, vecs in [
    ("bypass_lora (hidden)", bypass_vecs),
    ("with_lora  (hidden)", lora_vecs),
    ("logits     (10-dim)", logit_vecs),
]:
    intra, inter = compute_class_distances(vecs, labels)
    gap = intra.mean() - inter.mean()
    results[name] = {"intra": intra, "inter": inter, "gap": gap}
    print(f"\n{name}:")
    print(f"  Intra-class: {intra.mean():.4f} ± {intra.std():.4f}")
    print(f"  Inter-class: {inter.mean():.4f} ± {inter.std():.4f}")
    print(f"  Gap:         {gap:.4f}")

# ===============================
# ANALYSIS 3: Classification accuracy
# ===============================
print("\n=== Classification accuracy ===")
correct = 0
total = len(classifications)
conf_scores = []
wrong_cases = []

for i, (pid, series, img_path) in enumerate(images_data[:len(classifications)]):
    pred_label, conf = classifications[i]
    conf_scores.append(conf)
    img_name = os.path.basename(img_path)
    # ตรวจ series match
    if str(series) in str(pred_label) or str(pred_label) in str(series):
        correct += 1
    else:
        wrong_cases.append((pid, series, pred_label, conf))

print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
print(f"Mean confidence: {np.mean(conf_scores):.4f}")
if wrong_cases:
    print(f"\nWrong predictions (first 5):")
    for pid, true_s, pred_s, conf in wrong_cases[:5]:
        name = pid.replace("movex_chain_","").replace("movex_sprocket_","")
        print(f"  {name:25s} true={true_s:8s} pred={pred_s:8s} conf={conf:.3f}")

# ===============================
# PLOT 1: t-SNE (bypass vs lora hidden + logits)
# ===============================
print("\nGenerating plots...")

unique_labels = sorted(set(labels))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
color_map = {l: c for l, c in zip(unique_labels, colors)}

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for ax, vecs, title in [
    (axes[0], bypass_vecs, "A) base_model() — bypass LoRA\n(v1 style)"),
    (axes[1], lora_vecs,   "B) model() — LoRA hidden state\n(same as A if diff=0)"),
    (axes[2], logit_vecs,  "C) logits (10-dim)\n(LoRA classification space)"),
]:
    perp = min(15, len(vecs)-1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(vecs)

    for lbl in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(
            embedded[idx, 0], embedded[idx, 1],
            c=[color_map[lbl]], label=lbl,
            s=90, alpha=0.85, edgecolors='white', linewidths=0.5
        )

    intra, inter = compute_class_distances(vecs, labels)
    gap = intra.mean() - inter.mean()
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.text(0.05, 0.97, f'Gap={gap:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

plt.suptitle("ViT Embedding Space Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("tsne_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: tsne_comparison.png")

# ===============================
# PLOT 2: Similarity distribution
# ===============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, res) in zip(axes, results.items()):
    intra, inter = res["intra"], res["inter"]
    ax.hist(intra, bins=25, alpha=0.6, color='green', label=f'Intra (μ={intra.mean():.3f})')
    ax.hist(inter, bins=25, alpha=0.6, color='red',   label=f'Inter (μ={inter.mean():.3f})')
    ax.axvline(intra.mean(), color='darkgreen', linestyle='--', linewidth=2)
    ax.axvline(inter.mean(), color='darkred',   linestyle='--', linewidth=2)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.93, f'Gap: {res["gap"]:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle("Intra vs Inter Class Cosine Similarity Distribution", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("similarity_distribution.png", dpi=150, bbox_inches='tight')
print("Saved: similarity_distribution.png")

# ===============================
# PLOT 3: Heatmap (logits — most meaningful)
# ===============================
fig, ax = plt.subplots(figsize=(12, 10))

unique_pids = sorted(set(pids))
pid_short = [p.replace("movex_chain_","").replace("movex_sprocket_","") for p in unique_pids]

centroids = []
for pid in unique_pids:
    idx = [i for i, p in enumerate(pids) if p == pid]
    c = logit_vecs[idx].mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-8)
    centroids.append(c)

centroids = np.array(centroids)
sim_matrix = cosine_similarity(centroids)

sns.heatmap(
    sim_matrix,
    xticklabels=pid_short,
    yticklabels=pid_short,
    annot=True, fmt=".2f",
    cmap="RdYlGn", vmin=0.0, vmax=1.0,
    ax=ax, annot_kws={"size": 8}
)
ax.set_title("Product Centroid Similarity (logits space)\nHigh diagonal = good separation",
             fontsize=12, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig("logits_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved: logits_heatmap.png")

plt.show()
print("\nDone!")
print("Files:")
print("  tsne_comparison.png       — t-SNE: bypass vs lora hidden vs logits")
print("  similarity_distribution.png — intra/inter class distribution")
print("  logits_heatmap.png        — product similarity ใน logits space")