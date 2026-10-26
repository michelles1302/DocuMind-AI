# DOCUMIND AI 🧠

> **Tagline:** "Turn documents into knowledge." &nbsp;•&nbsp; *"Read less. Understand more."*

**DocuMind AI** transforms static documents (PDF, DOCX, TXT) into interactive, searchable, and queryable AI Document Intelligence dashboards. Designed to run **100% locally and CPU-friendly**, it requires **zero external API keys, paid services, or complex vector databases**.

---

## 🚀 Key Features & Capabilities

- **📄 Multi-Format Document Extraction**: Reliable text extraction for PDF, Microsoft Word (`.docx`), and plain text (`.txt`).
- **📸 Scanned PDF Detection**: Automatically detects image-only or scanned PDFs and alerts the user: *"This document appears to contain scanned/image-based pages. OCR is required to extract its text."* without fabricating summaries.
- **📝 Multi-Level Smart Summaries**:
  - **Executive Summary**: 2-3 sentence core overview.
  - **Key Points**: 5–10 structured takeaway bullet points.
  - **Section Summaries**: Breakdown by document section/heading.
  - **One-Minute Summary**: 60-second quick capture (~50 words).
  - **Detailed Summary**: Full structured analysis.
- **🏷️ Topic & Key Concept Extraction**: N-gram TF-IDF topic analysis with relative importance percentages + technical concept extraction mapping occurrences across document pages.
- **🕵️ Entity Intelligence & Structured Data**: Automatically categorizes People, Organizations, Datasets, Models, Technical terms, and extracts Dates, Numbers, and Measurements.
- **🔍 Local Semantic Search**: Search document passages by meaning (not just exact keywords) using `sentence-transformers/all-MiniLM-L6-v2`.
- **💬 Grounded "Ask the Document"**: Ask natural language questions and receive precise answers backed by **verified source passage evidence** (page numbers, section headings, and exact text quotes).
- **🛡️ Strict Answer Grounding (Zero Hallucination)**: If information cannot be retrieved from the document, DocuMind AI responds: *"I couldn't find enough information in this document to answer that."*
- **📊 Document Comparison**: Compare two documents side-by-side to analyze semantic similarity, overlapping topics, and unique insights.
- **📜 Persistent History**: Built-in SQLite database (`documind.db`) to store analyzed documents and Q&A logs.
- **📥 Downloadable PDF Reports**: Export professional PDF Document Intelligence reports using ReportLab.

---

## 🛠️ Architecture & NLP Pipeline

```
UPLOAD DOCUMENT (PDF / DOCX / TXT)
        │
        ▼
EXTRACT TEXT & CHECK SCANNED STATUS (pypdf / python-docx)
        │
        ▼
DETECT DOCUMENT STRUCTURE & SECTIONS
        │
        ▼
NLP ANALYSIS & FEATURE EXTRACTION
(Smart Summaries • Key Topics • Key Concepts • Entity Extraction)
        │
        ▼
LOCAL VECTOR STORE EMBEDDINGS (sentence-transformers/all-MiniLM-L6-v2)
        │
        ▼
INTERACTIVE STREAMLIT DASHBOARD & GROUNDED Q&A ENGINE
        │
        ▼
PERSIST TO SQLITE DB & EXPORT PDF REPORT
```

---

## 💻 Installation & Running Locally

### 1. Prerequisites
Python 3.10+ installed on Windows, macOS, or Linux.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Dashboard
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run CLI Summarizer
To run fast CLI document intelligence analysis directly in your terminal:
```bash
python summarize.py notes.txt
```

### 5. Run Automated Test Suite
To verify end-to-end functionality across PDF, DOCX, Scanned PDF, Q&A Grounding, PDF Reports, and Document Comparison:
```bash
python test_documind.py
```

---

## 📂 Project Structure

```
AI-Document Summarize/
├── app.py                  # Main Streamlit Dashboard Application UI
├── summarize.py            # Upgraded CLI Document Intelligence Script
├── test_documind.py        # Automated End-to-End Test Suite
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
└── src/                    # Modular Engine Package
    ├── __init__.py
    ├── extractor.py        # PDF, DOCX, TXT extraction & Scanned PDF detection
    ├── analyzer.py         # Smart summaries, topics, concepts, entities & structure
    ├── search_engine.py    # Local sentence-transformer vector store & grounded Q&A
    ├── database.py        # SQLite history persistence (documind.db)
    ├── reporter.py        # ReportLab PDF report generator
    └── comparator.py      # Document A vs Document B comparison engine
```

---

## 🔒 Limitations & Future Enhancements

- **Scanned OCR Support**: Scanned PDFs currently return an explicit notice requesting OCR. Tesseract OCR can be added as an optional plugin.
- **Extremely Large PDFs**: Embedded in-memory vector store is optimized for documents up to ~250 pages.

---

## 📜 License
MIT License - Open Source & Free to Use.
