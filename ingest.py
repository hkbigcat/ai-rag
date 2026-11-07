# -*- coding: utf-8 -*-
"""Phase 1c: embed chunks 並寫入 Pinecone。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pypdf import PdfReader
from ollama import Client

load_dotenv(Path(__file__).with_name(".env"))

PDF_PATH = Path(__file__).with_name("advice_household_disinfection_flu_chi.pdf")
EMBED_MODEL = "nomic-embed-text"
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "household_disinfection")
BATCH = 32
# 你環境變數 OLLAMA_HOST=0.0.0.0 是給 server 綁定用的，client 要連 127.0.0.1
ollama = Client(host="http://127.0.0.1:11434")


def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    docs = []
    for page_no, text in pages:
        for j, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                {
                    "id": f"p{page_no}-c{j}",
                    "text": chunk,
                    "page": page_no,
                    "source": PDF_PATH.name,
                }
            )
    return docs


def embed_texts(texts):
    vectors = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vectors.append(resp["embedding"])
    return vectors


def upsert_docs(docs):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception as exc:
        print(f"namespace clear skipped: {exc}")

    for start in range(0, len(docs), BATCH):
        batch = docs[start : start + BATCH]
        vectors = embed_texts([d["text"] for d in batch])
        payload = []
        for doc, vec in zip(batch, vectors):
            payload.append(
                {
                    "id": doc["id"],
                    "values": vec,
                    "metadata": {
                        "text": doc["text"],
                        "page": doc["page"],
                        "source": doc["source"],
                    },
                }
            )
        index.upsert(vectors=payload, namespace=NAMESPACE)
        print(f"upserted {start + len(batch)}/{len(docs)}")


def smoke_query(question, top_k=3):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    qvec = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    res = index.query(
        vector=qvec,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
    )
    print(f"\nQuery: {question}")
    for m in res.matches:
        meta = m.metadata or {}
        print(f"- score={m.score:.4f} page={meta.get('page')} id={m.id}")
        print((meta.get("text") or "")[:160].replace("\n", " "))
        print()


if __name__ == "__main__":
    pages = extract_pages(PDF_PATH)
    docs = chunk_pages(pages)
    print(f"pages={len(pages)} chunks={len(docs)} index={INDEX_NAME} ns={NAMESPACE}")
    upsert_docs(docs)
    smoke_query("流感高峰期，家居表面應該用什麼漂白水比例消毒？")
