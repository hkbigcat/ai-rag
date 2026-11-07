# -*- coding: utf-8 -*-
"""Phase 1a: 從 PDF 抽出純文字。"""
from pathlib import Path
from pypdf import PdfReader

PDF_PATH = Path(__file__).with_name("advice_household_disinfection_flu_chi.pdf")
OUT_PATH = Path(__file__).with_name("extracted_text.txt")


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # 頁碼標記方便之後引用與除錯
        parts.append(f"\n\n--- page {i + 1} ---\n{text}")
    return "".join(parts).strip()


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise SystemExit(f"找不到 PDF: {PDF_PATH}")
    text = extract_text(PDF_PATH)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"pages done, chars={len(text)}, saved={OUT_PATH.name}")
    print(text[:400])
