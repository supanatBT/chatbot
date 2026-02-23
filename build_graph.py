"""
build_graph.py
==============
Script สำหรับ build Knowledge Graph ลง SQLite

รันครั้งเดียวก่อนเริ่มใช้งาน chatbot หรือรันใหม่เมื่อ JSON data เปลี่ยน

Usage:
  python build_graph.py                        # ใช้ path default
  python build_graph.py --json path/to/data.json --db path/to/kg.db
  python build_graph.py --info                 # แค่ดู DB info ไม่ rebuild
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# ── Default paths ──────────────────────────────────────────────
DEFAULT_JSON = "./extracted_data_v3_machine_ready.json"
DEFAULT_DB   = "./movex_kg.db"


def build(json_path: str, db_path: str):
    """Build graph จาก JSON แล้ว save ลง SQLite"""
    from knowledge_graph import KnowledgeGraph

    print("=" * 60)
    print("  Movex Knowledge Graph Builder")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── โหลด JSON ──────────────────────────────────────────
    if not os.path.isfile(json_path):
        print(f"❌ ไม่พบ JSON: {json_path}")
        sys.exit(1)

    print(f"📂 โหลด JSON: {json_path}")
    t0 = time.perf_counter()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    product_db  = {p["product_id"]: p for p in data.get("products", []) if "product_id" in p}
    material_db = {m["material_code"]: m for m in data.get("Materials", [])}
    process_db  = {p["manufacturing_process"]: p for p in data.get("manufacturing_process", [])}

    print(
        f"   products={len(product_db)}  "
        f"materials={len(material_db)}  "
        f"processes={len(process_db)}  "
        f"({(time.perf_counter()-t0)*1000:.0f}ms)"
    )
    print()

    # ── Build + Save ────────────────────────────────────────
    kg = KnowledgeGraph.build_and_save(
        product_db=product_db,
        material_db=material_db,
        process_db=process_db,
        db_path=db_path,
    )

    # ── Verify ──────────────────────────────────────────────
    print()
    print("🔍 Verifying — โหลดกลับจาก DB เพื่อตรวจสอบ...")
    kg2   = KnowledgeGraph.load_from_db(db_path)
    info  = kg2.db_info()

    print()
    print("=" * 60)
    print("  ✅ Build สำเร็จ!")
    print(f"  📁 DB: {os.path.abspath(db_path)}")
    print(f"  🕐 Built at: {info.get('built_at')}")
    print(f"  📊 Nodes: {info.get('node_count')}  →  {info.get('node_types')}")
    print(f"  🔗 Edges: {info.get('edge_count')}  →  {info.get('edge_relations')}")
    print(f"  📋 Schema version: {info.get('schema_version')}")
    print()
    print("  ✨ พร้อมใช้งาน — chatbot_v3.py จะโหลดจาก DB นี้อัตโนมัติ")
    print("=" * 60)

    return kg


def show_info(db_path: str):
    """แสดง info ของ DB ที่มีอยู่โดยไม่ rebuild"""
    from knowledge_graph import KnowledgeGraph

    print(f"📋 DB Info: {db_path}")
    if not os.path.isfile(db_path):
        print("   ❌ ไม่พบไฟล์นี้ — กรุณารัน build_graph.py ก่อน")
        return

    kg   = KnowledgeGraph.load_from_db(db_path)
    info = kg.db_info()

    print(f"   Built at       : {info.get('built_at', 'unknown')}")
    print(f"   Schema version : {info.get('schema_version', 'unknown')}")
    print(f"   Nodes          : {info.get('node_count')}")
    print(f"   Node types     : {info.get('node_types')}")
    print(f"   Edges          : {info.get('edge_count')}")
    print(f"   Edge relations : {info.get('edge_relations')}")
    print(f"   File size      : {os.path.getsize(db_path) / 1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Movex Knowledge Graph ลง SQLite"
    )
    parser.add_argument(
        "--json", default=DEFAULT_JSON,
        help=f"Path ไปยัง JSON data (default: {DEFAULT_JSON})"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Path ไปยัง output SQLite DB (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="แค่ดู info ของ DB ที่มีอยู่ ไม่ rebuild"
    )
    args = parser.parse_args()

    if args.info:
        show_info(args.db)
    else:
        build(args.json, args.db)
