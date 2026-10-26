import numpy as np
import re
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = None

def load_embedding_model():
    """Load and cache local sentence-transformers model CPU-friendly."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL_CACHE


class DocumentVectorStore:
    """In-memory semantic vector store for a document."""

    def __init__(self, pages_data: list[dict], doc_structure: list[dict]):
        self.model = load_embedding_model()
        self.chunks = []
        self.chunk_embeddings = None
        self._build_chunks(pages_data, doc_structure)
        self._build_embeddings()

    def _build_chunks(self, pages_data: list[dict], doc_structure: list[dict]):
        """Split document into semantic passage chunks with page and section metadata."""
        chunk_id = 0
        for p in pages_data:
            page_num = p["page_num"]
            page_text = p["text"]
            if not page_text.strip():
                continue

            # Determine section name for this page
            section_name = f"Page {page_num}"
            if doc_structure:
                for sec in doc_structure:
                    if sec.get("page") == page_num:
                        section_name = sec["heading"]
                        break

            # Split page text into ~150-250 word paragraphs/chunks
            paragraphs = [para.strip() for para in page_text.split("\n\n") if para.strip()]
            if not paragraphs:
                paragraphs = [page_text]

            current_chunk = []
            current_words = 0

            for para in paragraphs:
                p_words = len(para.split())
                if current_words + p_words > 220 and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    self.chunks.append({
                        "chunk_id": chunk_id,
                        "page_num": page_num,
                        "section_name": section_name,
                        "text": chunk_text
                    })
                    chunk_id += 1
                    current_chunk = [para]
                    current_words = p_words
                else:
                    current_chunk.append(para)
                    current_words += p_words

            if current_chunk:
                chunk_text = " ".join(current_chunk)
                self.chunks.append({
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "section_name": section_name,
                    "text": chunk_text
                })
                chunk_id += 1

    def _build_embeddings(self):
        """Compute vector embeddings for all document chunks."""
        if not self.chunks:
            self.chunk_embeddings = np.array([])
            return

        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self.chunk_embeddings = np.array(embeddings)

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search document passages by semantic meaning."""
        if self.chunk_embeddings is None or len(self.chunks) == 0 or not query.strip():
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity (since vectors are normalized, dot product = cosine sim)
        similarities = np.dot(self.chunk_embeddings, query_vec)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            chunk = self.chunks[idx]
            results.append({
                "page_num": chunk["page_num"],
                "section_name": chunk["section_name"],
                "score": round(score * 100, 1),
                "text": chunk["text"]
            })
        return results

    def ask_document(self, question: str) -> dict:
        """
        Ask the document a question and return grounded answer + source evidence.
        Strict grounding rule: Never fabricate information if passages lack relevance.
        """
        if not question.strip():
            return {
                "answer": "Please enter a valid question.",
                "has_evidence": False,
                "evidence": []
            }

        search_results = self.semantic_search(question, top_k=3)

        # Non-hallucinating grounding threshold (score in %)
        if not search_results or search_results[0]["score"] < 15.0:
            return {
                "answer": "I couldn't find enough information in this document to answer that.",
                "has_evidence": False,
                "evidence": []
            }

        # Filter top evidence passages
        top_passages = [res for res in search_results if res["score"] >= 10.0]

        # Extract precise answer sentence(s) from top passage
        top_text = top_passages[0]["text"]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_text) if len(s.strip()) > 10]

        # Rank sentences within the top passage by query term matching & sentence length
        q_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
        best_sentences = []

        for s in sentences:
            s_words = set(re.sub(r"[^\w\s]", "", s.lower()).split())
            overlap = len(q_words.intersection(s_words))
            best_sentences.append((overlap, s))

        best_sentences.sort(key=lambda x: x[0], reverse=True)

        if best_sentences and best_sentences[0][0] > 0:
            # Pick top matching sentences
            answer_parts = [best_sentences[0][1]]
            if len(best_sentences) > 1 and best_sentences[1][0] > 0 and best_sentences[1][1] != best_sentences[0][1]:
                answer_parts.append(best_sentences[1][1])
            answer = " ".join(answer_parts)
        else:
            # Fallback to taking the first sentence or two of top passage
            answer = " ".join(sentences[:2]) if sentences else top_text[:200] + "..."

        evidence_list = []
        for p in top_passages:
            evidence_list.append({
                "page": p["page_num"],
                "section": p["section_name"],
                "score": p["score"],
                "text": p["text"]
            })

        return {
            "answer": answer,
            "has_evidence": True,
            "evidence": evidence_list
        }
