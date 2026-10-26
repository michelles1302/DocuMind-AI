import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = STOP_WORDS.union({
    "paper", "study", "author", "authors", "using", "used", "results", "table",
    "figure", "shown", "shows", "show", "based", "method", "methods", "approach",
    "section", "page", "data", "system", "proposed", "also", "one", "two", "three",
    "new", "well", "use", "may", "etc", "however", "thus", "provides", "present"
})


def tokenize_sentences(text: str) -> list[str]:
    """Tokenize text into clean sentences."""
    if not text.strip():
        return []
    sents = nltk.sent_tokenize(text)
    clean_sents = [re.sub(r"\s+", " ", s).strip() for s in sents if len(s.strip()) > 15]
    return clean_sents if clean_sents else [text.strip()]


def generate_smart_summaries(full_text: str, doc_structure: list[dict]) -> dict:
    """Generate multi-level summaries: Executive, Key Points, Section Summaries, One-Min, Detailed."""
    sentences = tokenize_sentences(full_text)

    if not sentences:
        empty = "No text available to summarize."
        return {
            "executive": empty,
            "key_points": [],
            "section_summaries": [],
            "one_minute": empty,
            "detailed": empty
        }

    if len(sentences) <= 4:
        full_summary = " ".join(sentences)
        return {
            "executive": full_summary,
            "key_points": sentences,
            "section_summaries": [{"heading": "Main Content", "summary": full_summary}],
            "one_minute": full_summary,
            "detailed": full_summary
        }

    # TF-IDF sentence scoring
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = tfidf_matrix.sum(axis=1).A1
    except Exception:
        # Fallback uniform scoring
        scores = [1.0] * len(sentences)

    # 1. Executive Summary (top 3 sentences preserving order)
    top_3_idx = sorted(scores.argsort()[-3:])
    executive_summary = " ".join([sentences[i] for i in top_3_idx])

    # 2. Key Points (5-8 top sentences)
    num_kp = min(8, max(5, len(sentences) // 4))
    top_kp_idx = sorted(scores.argsort()[-num_kp:])
    key_points = [sentences[i] for i in top_kp_idx]

    # 3. One-Minute Summary (top 2 most central sentences, ~40-60 words)
    top_2_idx = sorted(scores.argsort()[-2:])
    one_minute = " ".join([sentences[i] for i in top_2_idx])

    # 4. Detailed Summary (top 35% of sentences)
    num_det = max(6, int(len(sentences) * 0.35))
    top_det_idx = sorted(scores.argsort()[-num_det:])
    detailed_summary = " ".join([sentences[i] for i in top_det_idx])

    # 5. Section Summaries
    section_summaries = []
    if doc_structure:
        for sec in doc_structure:
            sec_sents = tokenize_sentences(sec["text"])
            if sec_sents:
                if len(sec_sents) <= 2:
                    sec_sum = " ".join(sec_sents)
                else:
                    try:
                        v = TfidfVectorizer(stop_words="english")
                        m = v.fit_transform(sec_sents)
                        sc = m.sum(axis=1).A1
                        top_i = sorted(sc.argsort()[-2:])
                        sec_sum = " ".join([sec_sents[i] for i in top_i])
                    except Exception:
                        sec_sum = sec_sents[0]
                section_summaries.append({
                    "heading": sec["heading"],
                    "page": sec.get("page", 1),
                    "summary": sec_sum
                })

    if not section_summaries:
        section_summaries.append({
            "heading": "Overview",
            "page": 1,
            "summary": executive_summary
        })

    return {
        "executive": executive_summary,
        "key_points": key_points,
        "section_summaries": section_summaries,
        "one_minute": one_minute,
        "detailed": detailed_summary
    }


def extract_key_topics(full_text: str, max_topics: int = 8) -> list[dict]:
    """Extract major document topics with relative importance using TF-IDF n-grams."""
    clean = re.sub(r"[^\w\s]", " ", full_text.lower())
    words = [w for w in clean.split() if w not in CUSTOM_STOPWORDS and len(w) > 2 and not w.isdigit()]

    if not words:
        return []

    # Use TfidfVectorizer for unigrams & bigrams
    try:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=list(CUSTOM_STOPWORDS),
            max_features=50
        )
        matrix = tfidf.fit_transform([full_text])
        feature_names = tfidf.get_feature_names_out()
        scores = matrix.toarray()[0]

        top_indices = scores.argsort()[::-1]
        topics = []

        max_score = scores[top_indices[0]] if len(top_indices) > 0 and scores[top_indices[0]] > 0 else 1.0

        seen_words = set()
        for idx in top_indices:
            term = feature_names[idx]
            # Ignore sub-phrases of already added terms
            if any(term in seen for seen in seen_words):
                continue

            score = scores[idx]
            if score <= 0:
                continue

            importance = int(round((score / max_score) * 100))
            if importance < 15:
                continue

            # Capitalize topic nicely
            formatted_topic = " ".join([w.capitalize() for w in term.split()])
            topics.append({
                "topic": formatted_topic,
                "importance": importance,
                "score": round(float(score), 4)
            })
            seen_words.add(term)

            if len(topics) >= max_topics:
                break

        return topics
    except Exception:
        # Fallback to Word Count
        counts = Counter(words).most_common(max_topics)
        if not counts:
            return []
        max_c = counts[0][1]
        return [{
            "topic": w.capitalize(),
            "importance": int((c / max_c) * 100),
            "score": float(c)
        } for w, c in counts]


def extract_key_concepts(full_text: str, pages_data: list[dict], max_concepts: int = 15) -> list[dict]:
    """Extract key technical/conceptual terms and locate their occurrences in pages."""
    # Pattern for technical terms: Titlecase phrases, acronyms, compound terms
    concept_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}(?:\-[A-Za-z0-9]+)?)\b"
    matches = re.findall(concept_pattern, full_text)

    # Filter out generic words
    filtered_matches = []
    for m in matches:
        lower = m.lower()
        if not any(w in CUSTOM_STOPWORDS for w in lower.split()) and len(m) > 3:
            filtered_matches.append(m)

    counts = Counter(filtered_matches).most_common(max_concepts * 2)

    concepts = []
    seen = set()

    for term, count in counts:
        if term.lower() in seen:
            continue

        # Find occurrences across pages
        occurrences = []
        for p in pages_data:
            if term.lower() in p["text"].lower():
                # Find snippet
                pos = p["text"].lower().find(term.lower())
                start = max(0, pos - 50)
                end = min(len(p["text"]), pos + len(term) + 60)
                snippet = "..." + p["text"][start:end].replace("\n", " ") + "..."
                occurrences.append({
                    "page": p["page_num"],
                    "snippet": snippet
                })

        if occurrences:
            concepts.append({
                "concept": term,
                "count": count,
                "occurrences": occurrences[:5]  # Top 5 page instances
            })
            seen.add(term.lower())

        if len(concepts) >= max_concepts:
            break

    return concepts


def extract_entities_and_info(full_text: str, pages_data: list[dict]) -> dict:
    """Extract structured entities (People, Organizations, Locations, Datasets, Models, Dates, Numbers, Measurements)."""
    # Regex patterns for structured info
    dates_pattern = r"\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:, \d{4})?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    percentages_pattern = r"\b\d+(?:\.\d+)?%\b"
    numbers_pattern = r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(?:images|patients|samples|layers|epochs|parameters|classes|features|accuracy|loss)\b"
    measurements_pattern = r"\b\d+\s*[×x]\s*\d+(?:\s*[×x]\s*\d+)?\b|\b(?:learning rate|lr|batch size|dropout)\s*=?\s*0?\.\d+|\b\d+\s*(?:ms|fps|GB|MB|kHz|MHz|GHz|mm|cm|px)\b"

    dates = sorted(list(set(re.findall(dates_pattern, full_text, re.IGNORECASE))))[:10]
    percentages = sorted(list(set(re.findall(percentages_pattern, full_text))))[:10]
    numbers = sorted(list(set(re.findall(numbers_pattern, full_text, re.IGNORECASE))))[:10]
    measurements = sorted(list(set(re.findall(measurements_pattern, full_text, re.IGNORECASE))))[:10]

    # Categorized Entity Heuristics
    org_keywords = ["university", "lab", "laboratory", "institute", "hospital", "corp", "inc", "group", "center", "centre", "department", "foundation", "google", "meta", "microsoft", "openai"]
    dataset_keywords = ["dataset", "data set", "corpus", "benchmark", "chestx", "imagenet", "mnist", "cifar", "coco", "mimic"]
    model_keywords = ["resnet", "cnn", "transformer", "bert", "vgg", "densenet", "yolo", "vit", "lstm", "gan", "u-net", "alexnet"]

    # Candidate capitalized phrases
    caps_matches = re.findall(r"\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)+\b", full_text)
    counts = Counter(caps_matches)

    people = []
    organizations = []
    datasets = []
    models = []
    locations = []
    technical_terms = []

    for term, c in counts.most_common(100):
        t_lower = term.lower()
        if any(w in CUSTOM_STOPWORDS for w in t_lower.split()):
            continue

        if any(k in t_lower for k in org_keywords) and term not in organizations:
            organizations.append(term)
        elif any(k in t_lower for k in dataset_keywords) and term not in datasets:
            datasets.append(term)
        elif any(k in t_lower for k in model_keywords) and term not in models:
            models.append(term)
        elif any(title in term for title in ["Dr.", "Prof.", "Mr.", "Ms.", "et al."]) or (len(term.split()) == 2 and not any(k in t_lower for k in ["model", "network", "data", "image", "table", "figure"])):
            if term not in people and len(term.split()) == 2:
                people.append(term)
        else:
            if len(term) > 4 and term not in technical_terms:
                technical_terms.append(term)

    # Attach occurrence snippets for entity breakdown lookup
    def get_entity_passages(entity_name):
        passages = []
        for p in pages_data:
            if entity_name.lower() in p["text"].lower():
                pos = p["text"].lower().find(entity_name.lower())
                start = max(0, pos - 40)
                end = min(len(p["text"]), pos + len(entity_name) + 60)
                passages.append({
                    "page": p["page_num"],
                    "snippet": "..." + p["text"][start:end].replace("\n", " ") + "..."
                })
        return passages[:3]

    entities_grouped = {
        "People": [{"name": p, "passages": get_entity_passages(p)} for p in people[:6]],
        "Organizations": [{"name": o, "passages": get_entity_passages(o)} for o in organizations[:6]],
        "Datasets": [{"name": d, "passages": get_entity_passages(d)} for d in datasets[:6]],
        "Models": [{"name": m, "passages": get_entity_passages(m)} for m in models[:6]],
        "Technical Terms": [{"name": t, "passages": get_entity_passages(t)} for t in technical_terms[:12]]
    }

    counts_summary = {
        "People": len(people),
        "Organizations": len(organizations),
        "Important Dates": len(dates),
        "Datasets": len(datasets),
        "Models": len(models),
        "Technical Concepts": len(technical_terms)
    }

    return {
        "counts": counts_summary,
        "entities": entities_grouped,
        "important_info": {
            "dates": dates,
            "percentages": percentages,
            "numbers": numbers,
            "measurements": measurements
        }
    }


def detect_document_structure(full_text: str, pages_data: list[dict]) -> list[dict]:
    """Detect document outline/headings and structure."""
    sections = []

    # Common section title patterns
    # e.g., 1. Introduction, Section 2: Methodology, ABSTRACT, CONCLUSION, ### Heading
    heading_pattern = r"^(?:\d+[\.\)]\s*|[A-Z\s]{4,25}$|#{1,4}\s+|[0-9]\.[0-9]\s+)?(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|Dataset|Model|Architecture|Experiments|Results|Discussion|Conclusion|References|Summary)"

    lines = full_text.split("\n")
    current_heading = "Introduction"
    current_lines = []
    current_page = 1

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        match = re.search(heading_pattern, line_clean, re.IGNORECASE)
        if match and len(line_clean) < 80:
            if current_lines:
                sec_text = "\n".join(current_lines)
                sections.append({
                    "heading": current_heading,
                    "page": current_page,
                    "text": sec_text,
                    "word_count": len(sec_text.split())
                })
            current_heading = line_clean.strip("#").strip()
            current_lines = []
            # Find page for this heading
            for p in pages_data:
                if line_clean in p["text"]:
                    current_page = p["page_num"]
                    break
        else:
            current_lines.append(line_clean)

    if current_lines:
        sec_text = "\n".join(current_lines)
        sections.append({
            "heading": current_heading,
            "page": current_page,
            "text": sec_text,
            "word_count": len(sec_text.split())
        })

    # Fallback if no explicit headings matched
    if not sections or len(sections) == 1:
        sections = []
        for p in pages_data:
            sections.append({
                "heading": f"Section / Page {p['page_num']}",
                "page": p["page_num"],
                "text": p["text"],
                "word_count": p["word_count"]
            })

    return sections


def generate_suggested_questions(topics: list[dict], entities: dict, doc_structure: list[dict]) -> list[str]:
    """Generate 6-8 smart, document-specific question suggestions."""
    questions = [
        "What is the main objective or purpose of this document?",
        "What are the core conclusions and takeaways?"
    ]

    # Check for methodology / architecture / dataset
    headings = [s["heading"].lower() for s in doc_structure]
    if any("method" in h or "architecture" in h or "approach" in h for h in headings):
        questions.append("What methodology or technical approach was implemented?")

    if any("dataset" in h or "data" in h for h in headings) or (entities.get("entities", {}).get("Datasets")):
        questions.append("What dataset or data source was used in this study?")

    if any("result" in h or "experiment" in h for h in headings):
        questions.append("What were the main experimental results and performance metrics?")

    # Add topic-based questions
    if topics:
        top_topic = topics[0]["topic"]
        questions.append(f"How does the document address {top_topic}?")
        if len(topics) > 1:
            second_topic = topics[1]["topic"]
            questions.append(f"What key findings are related to {second_topic}?")

    # Add entity-based question
    models = entities.get("entities", {}).get("Models", [])
    if models:
        questions.append(f"What is the performance of the {models[0]['name']} model?")

    # Deduplicate and return top 7
    unique_qs = []
    for q in questions:
        if q not in unique_qs:
            unique_qs.append(q)
    return unique_qs[:7]
