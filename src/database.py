import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "documind.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialize SQLite tables for history and document knowledge."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_type TEXT NOT NULL,
        upload_date TEXT NOT NULL,
        page_count INTEGER,
        word_count INTEGER,
        summary_json TEXT,
        topics_json TEXT,
        entities_json TEXT,
        structure_json TEXT,
        full_text TEXT,
        pages_json TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS qa_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        evidence_json TEXT,
        FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()

def save_document(filename: str, file_type: str, page_count: int, word_count: int,
                  summaries: dict, topics: list, entities: dict, structure: list,
                  full_text: str, pages: list) -> int:
    """Save an analyzed document into SQLite history."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO documents (
        filename, file_type, upload_date, page_count, word_count,
        summary_json, topics_json, entities_json, structure_json, full_text, pages_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, file_type, now_str, page_count, word_count,
        json.dumps(summaries), json.dumps(topics), json.dumps(entities),
        json.dumps(structure), full_text, json.dumps(pages)
    ))

    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def get_all_documents() -> list[dict]:
    """Retrieve list of all analyzed documents in history."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, filename, file_type, upload_date, page_count, word_count, summary_json, topics_json
    FROM documents ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for r in rows:
        summary_data = json.loads(r[6]) if r[6] else {}
        topics_data = json.loads(r[7]) if r[7] else []
        docs.append({
            "id": r[0],
            "filename": r[1],
            "file_type": r[2],
            "upload_date": r[3],
            "page_count": r[4],
            "word_count": r[5],
            "executive_summary": summary_data.get("executive", ""),
            "topics": [t["topic"] for t in topics_data[:4]]
        })
    return docs

def get_document(doc_id: int) -> dict:
    """Fetch complete document data by ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    r = cursor.fetchone()
    conn.close()

    if not r:
        return None

    return {
        "id": r[0],
        "filename": r[1],
        "file_type": r[2],
        "upload_date": r[3],
        "page_count": r[4],
        "word_count": r[5],
        "summaries": json.loads(r[6]) if r[6] else {},
        "topics": json.loads(r[7]) if r[7] else [],
        "entities": json.loads(r[8]) if r[8] else {},
        "structure": json.loads(r[9]) if r[9] else [],
        "full_text": r[10],
        "pages": json.loads(r[11]) if r[11] else []
    }

def delete_document(doc_id: int):
    """Delete document from database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    cursor.execute("DELETE FROM qa_history WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()

def save_qa_record(doc_id: int, question: str, answer: str, evidence: list):
    """Save Q&A record for history tracking."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO qa_history (doc_id, question, answer, timestamp, evidence_json)
    VALUES (?, ?, ?, ?, ?)
    """, (doc_id, question, answer, now_str, json.dumps(evidence)))

    conn.commit()
    conn.close()

def get_qa_history(doc_id: int) -> list[dict]:
    """Retrieve Q&A history for a document."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT question, answer, timestamp, evidence_json FROM qa_history
    WHERE doc_id = ? ORDER BY id DESC
    """, (doc_id,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "question": r[0],
            "answer": r[1],
            "timestamp": r[2],
            "evidence": json.loads(r[3]) if r[3] else []
        })
    return history
