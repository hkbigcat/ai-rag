# -*- coding: utf-8 -*-
"""Phase 2: 檢索相關段落 → 組 prompt → 用本機 qwen 回答。"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client
from pinecone import Pinecone

load_dotenv(Path(__file__).with_name(".env"))

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "household_disinfection")
TOP_K = 3

# client 必須連 127.0.0.1；系統 OLLAMA_HOST=0.0.0.0 不能拿來當 client 位址
ollama = Client(host="http://127.0.0.1:11434")


def retrieve(question: str, top_k: int = TOP_K):
    """問題 → embedding → Pinecone 相似度搜尋。"""
    t0 = time.perf_counter()
    qvec = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    embed_ms = (time.perf_counter() - t0) * 1000

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    t1 = time.perf_counter()
    res = index.query(
        vector=qvec,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
    )
    query_ms = (time.perf_counter() - t1) * 1000
    return res.matches, embed_ms, query_ms


def build_prompt(question: str, matches) -> str:
    """只根據檢索到的內容回答，減少胡謅。"""
    context_blocks = []
    for i, m in enumerate(matches, 1):
        meta = m.metadata or {}
        context_blocks.append(
            f"[段落{i} | page={meta.get('page')} | score={m.score:.4f}]\n"
            f"{meta.get('text', '')}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "你是衞生防護資訊助理。請只根據下列「參考段落」用繁體中文回答。\n"
        "若參考段落沒有答案，請明確說「文件中未有提及」，不要猜測。\n"
        "回答末列出你依據的段落編號。\n\n"
        f"參考段落：\n{context}\n\n"
        f"問題：{question}\n\n"
        "答案："
    )


def answer(question: str) -> None:
    matches, embed_ms, query_ms = retrieve(question)
    if not matches:
        print("沒有檢索到任何段落。請先跑 ingest.py。")
        return

    print("=== 檢索結果 ===")
    for i, m in enumerate(matches, 1):
        meta = m.metadata or {}
        preview = (meta.get("text") or "").replace("\n", " ")[:120]
        print(f"{i}. score={m.score:.4f} page={meta.get('page')} id={m.id}")
        print(f"   {preview}")

    prompt = build_prompt(question, matches)
    t2 = time.perf_counter()
    resp = ollama.generate(model=CHAT_MODEL, prompt=prompt)
    gen_ms = (time.perf_counter() - t2) * 1000

    print("\n=== 回答 ===")
    print(resp.get("response", "").strip())
    print("\n=== 延遲 (System Analyst 可監察) ===")
    print(f"embed_ms={embed_ms:.0f}  pinecone_ms={query_ms:.0f}  llm_ms={gen_ms:.0f}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "流感高峰期，家居表面應該用什麼漂白水比例消毒？"
    answer(q)
