import os
import sys
from src.extractor import extract_document
from src.analyzer import (
    generate_smart_summaries,
    extract_key_topics,
    extract_key_concepts,
    extract_entities_and_info,
    detect_document_structure
)

def run_cli_summarize(file_path: str = "notes.txt"):
    print("\n" + "=" * 50)
    print("      DOCUMIND AI - DOCUMENT INTELLIGENCE SYSTEM")
    print("            Turn documents into knowledge.")
    print("=" * 50 + "\n")

    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found.")
        sys.exit(1)

    print(f"Reading document: {file_path}...")
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    doc_raw = extract_document(file_bytes, os.path.basename(file_path))

    if doc_raw.get("is_scanned"):
        print("\n" + "!" * 50)
        print("WARNING: " + doc_raw.get("error"))
        print("!" * 50)
        sys.exit(1)

    if doc_raw.get("error"):
        print(f"\nERROR: {doc_raw.get('error')}")
        sys.exit(1)

    full_text = doc_raw["full_text"]
    pages_data = doc_raw["pages"]

    print(f"Extracted {doc_raw['page_count']} pages, {doc_raw['word_count']:,} words.")

    structure = detect_document_structure(full_text, pages_data)
    summaries = generate_smart_summaries(full_text, structure)
    topics = extract_key_topics(full_text)
    concepts = extract_key_concepts(full_text, pages_data)
    entities = extract_entities_and_info(full_text, pages_data)

    print("\n" + "=" * 50)
    print("1. EXECUTIVE SUMMARY")
    print("=" * 50)
    print(summaries["executive"])

    print("\n" + "=" * 50)
    print("2. KEY POINTS & TAKEAWAYS")
    print("=" * 50)
    for kp in summaries["key_points"][:8]:
        print(f" • {kp}")

    print("\n" + "=" * 50)
    print("3. MAJOR TOPICS")
    print("=" * 50)
    for t in topics:
        print(f" • {t['topic']} (Importance: {t['importance']}%)")

    print("\n" + "=" * 50)
    print("4. KEY TECHNICAL CONCEPTS")
    print("=" * 50)
    for c in concepts[:8]:
        print(f" • {c['concept']} (Appears {c['count']} times)")

    print("\n" + "=" * 50)
    print("5. DOCUMENT INSIGHTS & ENTITIES")
    print("=" * 50)
    for k, v in entities["counts"].items():
        print(f" • {k}: {v}")

    imp = entities.get("important_info", {})
    if imp.get("dates"):
        print(f"\nKey Dates: {', '.join(imp['dates'][:6])}")
    if imp.get("numbers"):
        print(f"Numbers & Metrics: {', '.join(imp['numbers'][:6])}")

    print("\n" + "=" * 50)
    print("6. DOCUMENT STRUCTURE OUTLINE")
    print("=" * 50)
    for sec in structure[:8]:
        print(f" Page {sec['page']}: {sec['heading']} ({sec['word_count']} words)")

    print("\nDocuMind AI Analysis Completed Successfully!")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notes.txt"
    run_cli_summarize(target)