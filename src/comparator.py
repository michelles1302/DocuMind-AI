import numpy as np
from src.search_engine import load_embedding_model


def compare_documents(doc_a: dict, doc_b: dict) -> dict:
    """Compare two analyzed documents and compute overlap, differences, and similarity."""
    # 1. Similarity score between full texts using sentence-transformers
    model = load_embedding_model()
    vec_a = model.encode([doc_a.get("full_text", "")[:2000]], normalize_embeddings=True)[0]
    vec_b = model.encode([doc_b.get("full_text", "")[:2000]], normalize_embeddings=True)[0]

    similarity_score = float(np.dot(vec_a, vec_b)) * 100.0
    similarity_score = max(0.0, min(100.0, similarity_score))

    # 2. Topic comparisons
    topics_a = {t["topic"].lower(): t["topic"] for t in doc_a.get("topics", [])}
    topics_b = {t["topic"].lower(): t["topic"] for t in doc_b.get("topics", [])}

    set_a = set(topics_a.keys())
    set_b = set(topics_b.keys())

    common_keys = set_a.intersection(set_b)
    unique_a_keys = set_a - set_b
    unique_b_keys = set_b - set_a

    common_topics = [topics_a[k] for k in common_keys]
    unique_topics_a = [topics_a[k] for k in unique_a_keys]
    unique_topics_b = [topics_b[k] for k in unique_b_keys]

    # 3. Concept overlaps
    concepts_a = {c["concept"].lower(): c["concept"] for c in doc_a.get("concepts", [])}
    concepts_b = {c["concept"].lower(): c["concept"] for c in doc_b.get("concepts", [])}

    common_concepts = [concepts_a[k] for k in set(concepts_a.keys()).intersection(set(concepts_b.keys()))]

    # 4. Summary metrics
    return {
        "doc_a_name": doc_a.get("filename", "Document A"),
        "doc_b_name": doc_b.get("filename", "Document B"),
        "similarity_percentage": round(similarity_score, 1),
        "doc_a_words": doc_a.get("word_count", 0),
        "doc_b_words": doc_b.get("word_count", 0),
        "doc_a_pages": doc_a.get("page_count", 1),
        "doc_b_pages": doc_b.get("page_count", 1),
        "doc_a_executive": doc_a.get("summaries", {}).get("executive", ""),
        "doc_b_executive": doc_b.get("summaries", {}).get("executive", ""),
        "common_topics": common_topics,
        "unique_topics_a": unique_topics_a,
        "unique_topics_b": unique_topics_b,
        "common_concepts": common_concepts
    }
