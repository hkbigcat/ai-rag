# -*- coding: utf-8 -*-
"""Phase 3: 簡單 Streamlit 問答介面。"""
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from ollama import Client
from pinecone import Pinecone

load_dotenv(Path(__file__).with_name(".env"))

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "household_disinfection")
TOP_K = 3

ollama = Client(host="http://127.0.0.1:11434")


def retrieve(question: str, top_k: int = TOP_K):
    t0 = time.perf_counter()
    qvec = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    embed_ms = (time.perf_counter() - t0) * 1000

    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(INDEX_NAME)
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
    blocks = []
    for i, m in enumerate(matches, 1):
        meta = m.metadata or {}
        blocks.append(
            f"[段落{i} | page={meta.get('page')} | score={m.score:.4f}]\n"
            f"{meta.get('text', '')}"
        )
    context = "\n\n".join(blocks)
    return (
        "你是衞生防護資訊助理。請只根據下列「參考段落」用繁體中文回答。\n"
        "若參考段落沒有答案，請明確說「文件中未有提及」，不要猜測。\n"
        "回答末列出你依據的段落編號。\n\n"
        f"參考段落：\n{context}\n\n"
        f"問題：{question}\n\n"
        "答案："
    )


def ask(question: str):
    matches, embed_ms, query_ms = retrieve(question)
    if not matches:
        return None, embed_ms, query_ms, 0.0

    prompt = build_prompt(question, matches)
    t2 = time.perf_counter()
    resp = ollama.generate(model=CHAT_MODEL, prompt=prompt)
    gen_ms = (time.perf_counter() - t2) * 1000
    return {
        "answer": (resp.get("response") or "").strip(),
        "matches": matches,
    }, embed_ms, query_ms, gen_ms


st.set_page_config(page_title="CHP 家居消毒 RAG", page_icon="🧼", layout="centered")
st.title("CHP 家居消毒 RAG")
st.caption(
    f"Embedding: `{EMBED_MODEL}` · Chat: `{CHAT_MODEL}` · "
    f"Index: `{INDEX_NAME}` / `{NAMESPACE}`"
)

question = st.text_input("輸入問題", placeholder="例如：家居表面漂白水比例是多少？")
ask_btn = st.button("提問", type="primary")

if ask_btn:
    q = (question or "").strip()
    if not q:
        st.warning("請先輸入問題。")
    else:
        with st.spinner("檢索中，接著本機 LLM 生成（可能要數十秒）…"):
            result, embed_ms, query_ms, gen_ms = ask(q)

        if result is None:
            st.error("沒有檢索到段落。請先執行 ingest.py。")
        else:
            st.subheader("回答")
            st.write(result["answer"])

            st.subheader("引用段落（請核對）")
            for i, m in enumerate(result["matches"], 1):
                meta = m.metadata or {}
                with st.expander(
                    f"段落 {i} · score={m.score:.4f} · page={meta.get('page')} · {m.id}",
                    expanded=(i == 1),
                ):
                    st.write(meta.get("text", ""))

            st.subheader("延遲")
            c1, c2, c3 = st.columns(3)
            c1.metric("embed_ms", f"{embed_ms:.0f}")
            c2.metric("pinecone_ms", f"{query_ms:.0f}")
            c3.metric("llm_ms", f"{gen_ms:.0f}")
