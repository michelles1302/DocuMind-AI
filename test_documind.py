import os
import sys
import docx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.extractor import extract_document
from src.analyzer import (
    generate_smart_summaries,
    extract_key_topics,
    extract_key_concepts,
    extract_entities_and_info,
    detect_document_structure,
    generate_suggested_questions
)
from src.search_engine import DocumentVectorStore
from src.reporter import generate_pdf_report
from src.comparator import compare_documents


def create_sample_docx(filepath):
    doc = docx.Document()
    doc.add_heading("Deep Learning in Medical Imaging", level=1)
    doc.add_paragraph("This paper presents a novel Convolutional Neural Network (CNN) architecture for chest X-ray classification.")
    doc.add_heading("Methodology and Dataset", level=2)
    doc.add_paragraph("The study utilized the ChestX-ray14 dataset containing 112,000 frontal chest X-ray images. The learning rate was set to 0.001 with a batch size of 32 across 50 epochs.")
    doc.add_heading("Results", level=2)
    doc.add_paragraph("The proposed ResNet-50 model achieved 94.2% binary classification accuracy on pneumonia detection, outperforming standard baseline models evaluated in January 2025.")
    doc.save(filepath)

def create_sample_pdf(filepath):
    c = canvas.Canvas(filepath, pagesize=letter)
    c.drawString(100, 750, "Research Report: Machine Learning Optimization")
    c.drawString(100, 730, "Author: Dr. Alex Rivera from Stanford Vision Lab.")
    c.drawString(100, 700, "1. Introduction")
    c.drawString(100, 680, "Machine learning algorithms require significant compute resources during training.")
    c.drawString(100, 650, "2. Experiments")
    c.drawString(100, 630, "We evaluated 10,000 images with 224 x 224 resolution and measured an average inference speed of 45 ms per image.")
    c.drawString(100, 600, "3. Conclusion")
    c.drawString(100, 580, "The optimized pipeline reduced memory footprint by 35% in 2026.")
    c.save()

def create_scanned_pdf(filepath):
    c = canvas.Canvas(filepath, pagesize=letter)
    # Draw shapes/images without any text elements
    c.rect(50, 50, 400, 600, fill=1)
    c.save()


def run_tests():
    print("=" * 60)
    print("        DOCUMIND AI - AUTOMATED END-TO-END TEST SUITE")
    print("=" * 60)

    # 1. Test DOCX
    docx_path = "test_paper.docx"
    create_sample_docx(docx_path)
    print(f"\n[TEST 1] Testing DOCX Extraction & NLP Pipeline on '{docx_path}'...")
    with open(docx_path, "rb") as f:
        docx_res = extract_document(f.read(), docx_path)

    assert not docx_res["is_scanned"], "DOCX should not be flagged as scanned"
    assert docx_res["word_count"] > 20, "Word count should be > 20"
    print(f" [OK] Extracted {docx_res['word_count']} words, {docx_res['page_count']} pages.")

    struct_docx = detect_document_structure(docx_res["full_text"], docx_res["pages"])
    sum_docx = generate_smart_summaries(docx_res["full_text"], struct_docx)
    topics_docx = extract_key_topics(docx_res["full_text"])
    concepts_docx = extract_key_concepts(docx_res["full_text"], docx_res["pages"])
    entities_docx = extract_entities_and_info(docx_res["full_text"], docx_res["pages"])

    assert sum_docx["executive"], "Executive summary should exist"
    assert len(topics_docx) > 0, "Key topics should be extracted"
    print(" [OK] Smart summaries, topics, concepts, and entities generated successfully.")

    # 2. Test PDF
    pdf_path = "test_paper.pdf"
    create_sample_pdf(pdf_path)
    print(f"\n[TEST 2] Testing PDF Extraction on '{pdf_path}'...")
    with open(pdf_path, "rb") as f:
        pdf_res = extract_document(f.read(), pdf_path)
    assert not pdf_res["is_scanned"], "Text PDF should not be flagged as scanned"
    assert pdf_res["word_count"] > 10, "PDF text extracted"
    print(f" [OK] PDF text extracted ({pdf_res['word_count']} words).")

    # 3. Test Scanned PDF Detection
    scanned_path = "test_scanned.pdf"
    create_scanned_pdf(scanned_path)
    print(f"\n[TEST 3] Testing Scanned PDF Detection on '{scanned_path}'...")
    with open(scanned_path, "rb") as f:
        scanned_res = extract_document(f.read(), scanned_path)
    assert scanned_res["is_scanned"], "Scanned PDF must be detected!"
    assert "OCR is required" in scanned_res["error"], "Correct scanned warning message returned!"
    print(f" [OK] Successfully caught scanned PDF: '{scanned_res['error']}'")

    # 4. Test Semantic Search & Ask Document Engine
    print("\n[TEST 4] Testing Semantic Vector Search & Grounded Q&A Engine...")
    vs = DocumentVectorStore(docx_res["pages"], struct_docx)

    # Valid question (present in doc)
    q_valid = "What dataset was used?"
    ans_valid = vs.ask_document(q_valid)
    assert ans_valid["has_evidence"], "Valid question must return evidence!"
    assert "ChestX-ray14" in ans_valid["answer"] or "ChestX-ray14" in ans_valid["evidence"][0]["text"], "Correct dataset answer retrieved!"
    print(f" [OK] Question: '{q_valid}' -> Answer: '{ans_valid['answer']}' (Evidence: Page {ans_valid['evidence'][0]['page']})")

    # Invalid question (NOT present in doc)
    q_invalid = "What is quantum mechanics entanglement and capital of Mars?"
    ans_invalid = vs.ask_document(q_invalid)
    assert not ans_invalid["has_evidence"] or "couldn't find enough information" in ans_invalid["answer"], "Missing info must not hallucinate!"
    print(f" [OK] Unanswerable Question -> Grounded Response: '{ans_invalid['answer']}'")

    # 5. Test PDF Report Generation
    print("\n[TEST 5] Testing PDF Report Generation...")
    doc_data = {
        "filename": "ChestXray_Study.docx",
        "file_type": "DOCX",
        "page_count": docx_res["page_count"],
        "word_count": docx_res["word_count"],
        "summaries": sum_docx,
        "topics": topics_docx,
        "entities": entities_docx,
        "structure": struct_docx
    }
    qa_hist = [{"question": q_valid, "answer": ans_valid["answer"], "evidence": ans_valid["evidence"]}]
    pdf_bytes = generate_pdf_report(doc_data, qa_hist)
    assert len(pdf_bytes) > 1000, "PDF report generated successfully"
    print(f" [OK] Generated PDF report ({len(pdf_bytes):,} bytes).")

    # 6. Test Document Comparison
    print("\n[TEST 6] Testing Document Comparison (Doc A vs Doc B)...")
    comp_res = compare_documents(doc_data, doc_data)
    assert comp_res["similarity_percentage"] > 90.0, "Identical docs should have high similarity"
    print(f" [OK] Comparison similarity index: {comp_res['similarity_percentage']}%")

    # Clean up test artifacts
    for file_to_rem in [docx_path, pdf_path, scanned_path]:
        if os.path.exists(file_to_rem):
            os.remove(file_to_rem)

    print("\n" + "=" * 60)
    print(" ALL 6 TEST SUITES PASSED PERFECTLY WITH ZERO ERRORS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
