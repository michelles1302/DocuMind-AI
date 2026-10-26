import streamlit as st
import json
import re
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
from src.database import (
    init_db,
    save_document,
    get_all_documents,
    get_document,
    delete_document,
    save_qa_record,
    get_qa_history
)
from src.reporter import generate_pdf_report
from src.comparator import compare_documents


# ==========================================
# PAGE CONFIG & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="DocuMind AI | Document Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for modern AI-product aesthetics
st.markdown("""
<style>
    /* Main container theme */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Top Header */
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #4F46E5;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #C7D2FE;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Topic Pill Tags */
    .topic-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 4px;
    }

    /* Q&A Box */
    .qa-box {
        background: #1E293B;
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 18px;
        margin-top: 14px;
    }
    .qa-answer {
        font-size: 1.05rem;
        color: #F8FAFC;
        line-height: 1.6;
    }

    /* Evidence Box */
    .evidence-box {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #CBD5E1;
    }
    .evidence-tag {
        color: #38BDF8;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;

    /* Hide standard Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize SQLite Database
init_db()


# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "selected_concept" not in st.session_state:
    st.session_state.selected_concept = None
if "active_question" not in st.session_state:
    st.session_state.active_question = ""


# ==========================================
# HEADER DISPLAY
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>DOCUMIND AI</h1>
    <p>Turn documents into knowledge. &nbsp;•&nbsp; <i>Read less. Understand more.</i></p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR NAVIGATION & FILE UPLOAD
# ==========================================
with st.sidebar:
    st.title("🧠 DocuMind Central")

    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "doc", "txt", "md"],
        help="Supports research papers, technical reports, and notes."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        # Trigger analysis if new file or different file uploaded
        if st.session_state.current_doc is None or st.session_state.current_doc.get("filename") != filename:
            with st.spinner("🔍 Extracting and Analyzing Document..."):
                doc_raw = extract_document(file_bytes, filename)

                if doc_raw.get("is_scanned"):
                    st.error(f"⚠️ {doc_raw.get('error')}")
                    st.session_state.current_doc = None
                    st.session_state.vector_store = None
                elif doc_raw.get("error"):
                    st.error(f"❌ {doc_raw.get('error')}")
                    st.session_state.current_doc = None
                    st.session_state.vector_store = None
                else:
                    full_text = doc_raw["full_text"]
                    pages_data = doc_raw["pages"]

                    # Run NLP Pipeline
                    structure = detect_document_structure(full_text, pages_data)
                    summaries = generate_smart_summaries(full_text, structure)
                    topics = extract_key_topics(full_text)
                    concepts = extract_key_concepts(full_text, pages_data)
                    entities = extract_entities_and_info(full_text, pages_data)
                    suggested_qs = generate_suggested_questions(topics, entities, structure)

                    doc_data = {
                        "filename": filename,
                        "file_type": doc_raw["file_type"],
                        "page_count": doc_raw["page_count"],
                        "word_count": doc_raw["word_count"],
                        "full_text": full_text,
                        "pages": pages_data,
                        "structure": structure,
                        "summaries": summaries,
                        "topics": topics,
                        "concepts": concepts,
                        "entities": entities,
                        "suggested_questions": suggested_qs
                    }

                    # Build Vector Store
                    vs = DocumentVectorStore(pages_data, structure)

                    # Save to DB
                    doc_id = save_document(
                        filename, doc_raw["file_type"], doc_raw["page_count"], doc_raw["word_count"],
                        summaries, topics, entities, structure, full_text, pages_data
                    )
                    doc_data["id"] = doc_id

                    st.session_state.current_doc = doc_data
                    st.session_state.vector_store = vs
                    st.success(f"✅ Analyzed: {filename}")

    st.markdown("---")

    # Navigation menu
    nav_option = st.radio(
        "Navigation",
        [
            "🚀 Document Dashboard",
            "💬 Ask the Document",
            "🔍 Semantic Search",
            "📑 Structure & Viewer",
            "📊 Compare Documents",
            "📜 History & Database",
            "ℹ️ About"
        ]
    )


# ==========================================
# MAIN CONTENT ROUTING
# ==========================================
current_doc = st.session_state.current_doc
vector_store = st.session_state.vector_store


if current_doc is None and nav_option not in ["📊 Compare Documents", "📜 History & Database", "ℹ️ About"]:
    st.info("👈 Please upload a document (.pdf, .docx, .txt) in the sidebar to begin Document Intelligence analysis.")

    # Welcome Banner / Feature Highlights
    st.markdown("""
    ### Welcome to DocuMind AI
    **DocuMind AI** transforms static documents into interactive knowledge structures.

    #### Key Capabilities:
    1. 📊 **Smart Summaries**: Executive, Key Points, Section breakdowns, 1-minute overview.
    2. 🏷️ **Key Topics & Concepts**: Automatic TF-IDF & phrase extraction with occurrence mapping.
    3. 🕵️ **Entity Intelligence**: Extract People, Organizations, Datasets, Models, Dates & Metrics.
    4. 🔍 **Semantic Search**: Query document passages by meaning via local `all-MiniLM-L6-v2` embeddings.
    5. 💬 **Grounded Ask-the-Document**: Get exact answers backed by verified source passage evidence.
    6. 📄 **Export Reports**: Generate downloadable PDF Intelligence Reports instantly.
    """)

elif nav_option == "🚀 Document Dashboard" and current_doc:
    st.subheader(f"📄 Overview: {current_doc['filename']}")

    # Top Metric Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{current_doc["page_count"]}</div><div class="metric-label">Pages</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{current_doc["word_count"]:,}</div><div class="metric-label">Words</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(current_doc["structure"])}</div><div class="metric-label">Sections</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(current_doc["topics"])}</div><div class="metric-label">Key Topics</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Export PDF Report Button
    pdf_bytes = generate_pdf_report(current_doc, get_qa_history(current_doc.get("id", 0)))
    st.download_button(
        label="📥 Export PDF Intelligence Report",
        data=pdf_bytes,
        file_name=f"DocuMind_Report_{current_doc['filename']}.pdf",
        mime="application/pdf",
        help="Download formatted PDF summary report"
    )

    st.markdown("---")

    # SMART SUMMARY TABS
    st.markdown("### 📝 Smart Summary")
    sum_tabs = st.tabs(["Executive Summary", "Key Points", "Section Summaries", "1-Minute Overview", "Detailed Summary"])

    summaries = current_doc["summaries"]

    with sum_tabs[0]:
        st.info(summaries.get("executive", "N/A"))

    with sum_tabs[1]:
        for kp in summaries.get("key_points", []):
            st.write(f"• {kp}")

    with sum_tabs[2]:
        for sec_sum in summaries.get("section_summaries", []):
            with st.expander(f"📌 {sec_sum['heading']} (Page {sec_sum.get('page', 1)})", expanded=True):
                st.write(sec_sum["summary"])

    with sum_tabs[3]:
        st.success(summaries.get("one_minute", "N/A"))

    with sum_tabs[4]:
        st.write(summaries.get("detailed", "N/A"))

    st.markdown("---")

    # KEY TOPICS & CONCEPTS
    c_top, c_con = st.columns([1, 1])

    with c_top:
        st.markdown("### 🏷️ Key Topics & Relative Importance")
        topics = current_doc.get("topics", [])
        for t in topics:
            col_t, col_p = st.columns([2, 3])
            with col_t:
                st.write(f"**{t['topic']}**")
            with col_p:
                st.progress(t["importance"] / 100.0, text=f"{t['importance']}%")

    with c_con:
        st.markdown("### 💡 Key Technical Concepts")
        concepts = current_doc.get("concepts", [])
        if concepts:
            concept_names = [c["concept"] for c in concepts]
            selected = st.selectbox("Click concept to view document occurrences:", concept_names)
            if selected:
                c_data = next((c for c in concepts if c["concept"] == selected), None)
                if c_data:
                    st.caption(f"Appears {c_data['count']} times in document:")
                    for occ in c_data["occurrences"]:
                        st.markdown(f"**Page {occ['page']}:** `{occ['snippet']}`")
        else:
            st.write("No specific multi-word technical concepts detected.")

    st.markdown("---")

    # ENTITY INTELLIGENCE & STRUCTURED INFO
    st.markdown("### 🕵️ Entity & Structured Information")
    entities = current_doc.get("entities", {})
    counts = entities.get("counts", {})

    if counts:
        e_cols = st.columns(len(counts))
        for idx, (k, v) in enumerate(counts.items()):
            with e_cols[idx]:
                st.metric(k, v)

    st.markdown("<br>", unsafe_allow_html=True)
    imp_info = entities.get("important_info", {})
    i_col1, i_col2, i_col3 = st.columns(3)

    with i_col1:
        st.markdown("**📅 Extracted Dates**")
        dates = imp_info.get("dates", [])
        if dates:
            for d in dates[:6]:
                st.write(f"• {d}")
        else:
            st.caption("None detected")

    with i_col2:
        st.markdown("**📊 Numbers & Percentages**")
        nums = imp_info.get("numbers", [])
        if nums:
            for n in nums[:6]:
                st.write(f"• {n}")
        else:
            st.caption("None detected")

    with i_col3:
        st.markdown("**📏 Measurements & Parameters**")
        meas = imp_info.get("measurements", [])
        if meas:
            for m in meas[:6]:
                st.write(f"• {m}")
        else:
            st.caption("None detected")

    st.markdown("---")

    # QUESTION SUGGESTIONS
    st.markdown("### ❓ Suggested Questions for this Document")
    suggested_qs = current_doc.get("suggested_questions", [])
    q_cols = st.columns(2)
    for idx, q in enumerate(suggested_qs):
        with q_cols[idx % 2]:
            if st.button(f"💬 {q}", key=f"sug_q_{idx}"):
                st.session_state.active_question = q
                st.info(f"Question copied! Go to 'Ask the Document' tab or run query: {q}")


elif nav_option == "💬 Ask the Document" and current_doc:
    st.subheader("💬 Ask the Document")
    st.caption("Get direct answers grounded strictly in document evidence. Non-fabricated assurance.")

    # Preset question buttons
    st.write("**Quick Questions:**")
    suggested_qs = current_doc.get("suggested_questions", [])
    btn_cols = st.columns(len(suggested_qs[:4]))
    for idx, sq in enumerate(suggested_qs[:4]):
        with btn_cols[idx]:
            if st.button(f"Q{idx+1}: {sq[:25]}...", key=f"ask_preset_{idx}"):
                st.session_state.active_question = sq

    question_input = st.text_input(
        "Ask any question about this document:",
        value=st.session_state.active_question,
        placeholder="e.g., What dataset was used in this study?"
    )

    if st.button("🔎 Ask Document", type="primary"):
        if question_input:
            with st.spinner("Retrieving evidence passages and formulating answer..."):
                res = vector_store.ask_document(question_input)

                st.markdown("### Answer")
                if res["has_evidence"]:
                    st.markdown(f'<div class="qa-box"><div class="qa-answer">{res["answer"]}</div></div>', unsafe_allow_html=True)

                    st.markdown("### 📌 Verified Source Evidence")
                    for ev in res["evidence"]:
                        st.markdown(f"""
                        <div class="evidence-box">
                            <span class="evidence-tag">Page {ev['page']} • {ev['section']} (Match Score: {ev['score']}%)</span><br>
                            "{ev['text']}"
                        </div>
                        """, unsafe_allow_html=True)

                    # Save Q&A to database
                    if current_doc.get("id"):
                        save_qa_record(current_doc["id"], question_input, res["answer"], res["evidence"])
                else:
                    st.warning(f"⚠️ {res['answer']}")

    # Past Q&A history for this document
    if current_doc.get("id"):
        qa_hist = get_qa_history(current_doc["id"])
        if qa_hist:
            st.markdown("---")
            st.markdown("### 📜 Q&A History for this Document")
            for item in qa_hist:
                with st.expander(f"Q: {item['question']} ({item['timestamp']})"):
                    st.write(f"**Answer:** {item['answer']}")
                    if item.get("evidence"):
                        st.caption(f"Source Evidence (Page {item['evidence'][0]['page']}): \"{item['evidence'][0]['text'][:150]}...\"")


elif nav_option == "🔍 Semantic Search" and current_doc:
    st.subheader("🔍 Semantic Document Search")
    st.caption("Search by meaning rather than exact keyword matches.")

    search_query = st.text_input("Enter search query or concept:", placeholder="e.g. dataset sample size or accuracy evaluation")

    if search_query:
        with st.spinner("Searching document vector space..."):
            results = vector_store.semantic_search(search_query, top_k=6)

            st.write(f"Found {len(results)} relevant passages:")
            for res in results:
                st.markdown(f"""
                <div class="evidence-box">
                    <span class="evidence-tag">Page {res['page_num']} • {res['section_name']} | Semantic Score: {res['score']}%</span><br>
                    {res['text']}
                </div>
                """, unsafe_allow_html=True)


elif nav_option == "📑 Structure & Viewer" and current_doc:
    st.subheader("📑 Document Structure & Full Text Viewer")

    col_out, col_txt = st.columns([1, 2])

    with col_out:
        st.markdown("### Table of Contents Outline")
        structure = current_doc.get("structure", [])
        for sec in structure:
            st.write(f"• **Page {sec['page']}:** {sec['heading']} ({sec['word_count']} words)")

    with col_txt:
        st.markdown("### Document Text Inspector")
        pages = current_doc.get("pages", [])
        if pages:
            selected_page = st.selectbox("Select Page:", [p["page_num"] for p in pages])
            page_data = next((p for p in pages if p["page_num"] == selected_page), None)
            if page_data:
                st.text_area(f"Page {selected_page} Text", page_data["text"], height=450)


elif nav_option == "📊 Compare Documents":
    st.subheader("📊 Compare Two Documents")
    st.caption("Analyze commonalities, differences, and topic overlaps between two documents.")

    col_a, col_b = st.columns(2)

    with col_a:
        file_a = st.file_uploader("Upload Document A", type=["pdf", "docx", "txt"], key="doc_comp_a")
    with col_b:
        file_b = st.file_uploader("Upload Document B", type=["pdf", "docx", "txt"], key="doc_comp_b")

    if file_a and file_b:
        if st.button("⚡ Run Document Comparison", type="primary"):
            with st.spinner("Analyzing and comparing both documents..."):
                res_a = extract_document(file_a.read(), file_a.name)
                res_b = extract_document(file_b.read(), file_b.name)

                if res_a.get("error") or res_b.get("error"):
                    st.error("Error reading one of the comparison documents.")
                else:
                    struct_a = detect_document_structure(res_a["full_text"], res_a["pages"])
                    struct_b = detect_document_structure(res_b["full_text"], res_b["pages"])

                    doc_a_dict = {
                        "filename": file_a.name,
                        "full_text": res_a["full_text"],
                        "page_count": res_a["page_count"],
                        "word_count": res_a["word_count"],
                        "summaries": generate_smart_summaries(res_a["full_text"], struct_a),
                        "topics": extract_key_topics(res_a["full_text"]),
                        "concepts": extract_key_concepts(res_a["full_text"], res_a["pages"])
                    }

                    doc_b_dict = {
                        "filename": file_b.name,
                        "full_text": res_b["full_text"],
                        "page_count": res_b["page_count"],
                        "word_count": res_b["word_count"],
                        "summaries": generate_smart_summaries(res_b["full_text"], struct_b),
                        "topics": extract_key_topics(res_b["full_text"]),
                        "concepts": extract_key_concepts(res_b["full_text"], res_b["pages"])
                    }

                    comp_res = compare_documents(doc_a_dict, doc_b_dict)

                    st.markdown("### Comparison Results")
                    st.metric("Semantic Similarity Index", f"{comp_res['similarity_percentage']}%")

                    st.markdown("---")

                    col_res_a, col_res_b = st.columns(2)
                    with col_res_a:
                        st.markdown(f"#### 📄 {comp_res['doc_a_name']}")
                        st.caption(f"{comp_res['doc_a_pages']} Pages | {comp_res['doc_a_words']:,} Words")
                        st.write(comp_res['doc_a_executive'])
                        st.markdown("**Unique Topics:**")
                        for ut in comp_res['unique_topics_a']:
                            st.write(f"• {ut}")

                    with col_res_b:
                        st.markdown(f"#### 📄 {comp_res['doc_b_name']}")
                        st.caption(f"{comp_res['doc_b_pages']} Pages | {comp_res['doc_b_words']:,} Words")
                        st.write(comp_res['doc_b_executive'])
                        st.markdown("**Unique Topics:**")
                        for ut in comp_res['unique_topics_b']:
                            st.write(f"• {ut}")

                    st.markdown("---")
                    st.markdown("#### 🤝 Overlapping Topics & Concepts")
                    if comp_res["common_topics"]:
                        st.write("**Common Topics:** " + ", ".join(comp_res["common_topics"]))
                    else:
                        st.write("No direct topic overlap detected.")

                    if comp_res["common_concepts"]:
                        st.write("**Shared Concepts:** " + ", ".join(comp_res["common_concepts"]))


elif nav_option == "📜 History & Database":
    st.subheader("📜 History & Past Document Analyses")
    docs_history = get_all_documents()

    if not docs_history:
        st.info("No documents stored in history database yet.")
    else:
        for dh in docs_history:
            with st.expander(f"📄 {dh['filename']} ({dh['upload_date']}) - {dh['page_count']} Pages"):
                st.write(f"**Executive Summary:** {dh['executive_summary']}")
                if dh.get("topics"):
                    st.write("**Key Topics:** " + ", ".join(dh["topics"]))

                c_load, c_del = st.columns([1, 1])
                with c_load:
                    if st.button("📂 Load Document Knowledge", key=f"load_db_{dh['id']}"):
                        full_doc_data = get_document(dh["id"])
                        if full_doc_data:
                            st.session_state.current_doc = full_doc_data
                            st.session_state.vector_store = DocumentVectorStore(
                                full_doc_data["pages"], full_doc_data["structure"]
                            )
                            st.success(f"Loaded: {dh['filename']}")
                with c_del:
                    if st.button("🗑️ Delete", key=f"del_db_{dh['id']}"):
                        delete_document(dh["id"])
                        st.warning("Deleted document from history.")
                        st.rerun()


elif nav_option == "ℹ️ About":
    st.subheader("ℹ️ About DocuMind AI")
    st.markdown("""
    **DocuMind AI** is an advanced AI Document Intelligence System designed to extract, analyze, structure, and query documents offline and CPU-friendly.

    ### 🛠️ Architecture & Technologies:
    - **Text Extraction**: `pypdf` (PDF), `python-docx` (DOCX), UTF-8 text processing.
    - **Scanned PDF Detection**: Rejects non-text image PDFs without fabricating summaries.
    - **NLP Pipeline**: TF-IDF, NLTK sentence tokenization, custom domain stopword filtering, regex entity extraction.
    - **Semantic Engine**: `sentence-transformers/all-MiniLM-L6-v2` embedding vector space.
    - **Database**: SQLite3 (`documind.db`) history and Q&A persistence.
    - **PDF Reporter**: `ReportLab` formatted PDF intelligence report generation.
    - **Grounding Rule**: Strict non-hallucinating evidence verification.
    """)