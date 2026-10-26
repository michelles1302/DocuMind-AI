import io
import re
import pypdf
import docx


def clean_text(text: str) -> str:
    """Clean and normalize raw extracted text."""
    if not text:
        return ""
    # Normalize multiple whitespace, tabs, and linebreaks
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_from_pdf(file_bytes: bytes) -> dict:
    """Extract text page by page from PDF and check for scanned content."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        pages_data = []
        full_text_list = []
        total_words = 0

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned_page = clean_text(page_text)
            words = len(cleaned_page.split())
            total_words += words
            pages_data.append({
                "page_num": idx + 1,
                "text": cleaned_page,
                "word_count": words
            })
            if cleaned_page:
                full_text_list.append(cleaned_page)

        full_text = "\n\n".join(full_text_list).strip()

        # Scanned PDF Detection Rule:
        # If document has pages but total extracted text is negligible
        if num_pages > 0 and (len(full_text) < 50 or (total_words / num_pages) < 5):
            return {
                "is_scanned": True,
                "error": "This document appears to contain scanned/image-based pages. OCR is required to extract its text.",
                "file_type": "PDF",
                "page_count": num_pages,
                "word_count": total_words,
                "full_text": "",
                "pages": []
            }

        return {
            "is_scanned": False,
            "error": None,
            "file_type": "PDF",
            "page_count": num_pages,
            "word_count": total_words,
            "full_text": full_text,
            "pages": pages_data
        }
    except Exception as e:
        return {
            "is_scanned": False,
            "error": f"Failed to extract PDF: {str(e)}",
            "file_type": "PDF",
            "page_count": 0,
            "word_count": 0,
            "full_text": "",
            "pages": []
        }


def extract_from_docx(file_bytes: bytes) -> dict:
    """Extract text from Microsoft Word DOCX files."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs).strip()
        cleaned = clean_text(full_text)
        words = len(cleaned.split())

        # Group paragraphs into pages of ~300 words for consistent chunking
        pages_data = []
        current_page_words = []
        current_word_count = 0
        page_num = 1

        for p in paragraphs:
            p_clean = clean_text(p)
            p_words = len(p_clean.split())
            if current_word_count + p_words > 350 and current_page_words:
                page_text = "\n\n".join(current_page_words)
                pages_data.append({
                    "page_num": page_num,
                    "text": page_text,
                    "word_count": len(page_text.split())
                })
                page_num += 1
                current_page_words = [p_clean]
                current_word_count = p_words
            else:
                current_page_words.append(p_clean)
                current_word_count += p_words

        if current_page_words:
            page_text = "\n\n".join(current_page_words)
            pages_data.append({
                "page_num": page_num,
                "text": page_text,
                "word_count": len(page_text.split())
            })

        return {
            "is_scanned": False,
            "error": None,
            "file_type": "DOCX",
            "page_count": len(pages_data) if pages_data else 1,
            "word_count": words,
            "full_text": cleaned,
            "pages": pages_data
        }
    except Exception as e:
        return {
            "is_scanned": False,
            "error": f"Failed to extract DOCX: {str(e)}",
            "file_type": "DOCX",
            "page_count": 0,
            "word_count": 0,
            "full_text": "",
            "pages": []
        }


def extract_from_txt(file_bytes: bytes) -> dict:
    """Extract text from plain text file with encoding fallbacks."""
    raw_text = ""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            raw_text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    cleaned = clean_text(raw_text)
    words = len(cleaned.split())

    if not cleaned:
        return {
            "is_scanned": False,
            "error": "The uploaded text file is empty.",
            "file_type": "TXT",
            "page_count": 0,
            "word_count": 0,
            "full_text": "",
            "pages": []
        }

    # Split into virtual pages by paragraphs or ~350 words
    paragraphs = [p for p in cleaned.split("\n\n") if p.strip()]
    pages_data = []
    current_page_words = []
    current_word_count = 0
    page_num = 1

    for p in paragraphs:
        p_words = len(p.split())
        if current_word_count + p_words > 350 and current_page_words:
            page_text = "\n\n".join(current_page_words)
            pages_data.append({
                "page_num": page_num,
                "text": page_text,
                "word_count": len(page_text.split())
            })
            page_num += 1
            current_page_words = [p]
            current_word_count = p_words
        else:
            current_page_words.append(p)
            current_word_count += p_words

    if current_page_words:
        page_text = "\n\n".join(current_page_words)
        pages_data.append({
            "page_num": page_num,
            "text": page_text,
            "word_count": len(page_text.split())
        })

    return {
        "is_scanned": False,
        "error": None,
        "file_type": "TXT",
        "page_count": len(pages_data),
        "word_count": words,
        "full_text": cleaned,
        "pages": pages_data
    }


def extract_document(file_bytes: bytes, filename: str) -> dict:
    """Main extraction router for supported document formats."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if ext == "pdf":
        result = extract_from_pdf(file_bytes)
    elif ext in ["docx", "doc"]:
        result = extract_from_docx(file_bytes)
    elif ext in ["txt", "md", "csv", "log"]:
        result = extract_from_txt(file_bytes)
    else:
        # Attempt UTF-8 txt fallback
        result = extract_from_txt(file_bytes)

    result["filename"] = filename
    return result
