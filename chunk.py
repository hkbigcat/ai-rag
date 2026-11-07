# -*- coding: utf-8 -*-
"""Phase 1b: 把長文切成重疊的小段（chunks）。"""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_PATH = Path(__file__).with_name("extracted_text.txt")
OUT_PATH = Path(__file__).with_name("chunks.txt")

# chunk_size: 每段大約多少字元；太大→檢索太粗，太小→失去上下文
# chunk_overlap: 相鄰段重疊，避免句子被切斷後找不到
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)


if __name__ == "__main__":
    full_text = TEXT_PATH.read_text(encoding="utf-8")
    chunks = splitter.split_text(full_text)
    OUT_PATH.write_text(
        "\n\n===== CHUNK =====\n\n".join(chunks),
        encoding="utf-8",
    )
    print(f"Created {len(chunks)} chunks -> {OUT_PATH.name}")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- preview chunk {i} ({len(c)} chars) ---\n{c[:180]}")
