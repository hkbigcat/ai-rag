# CHP Household Disinfection PDF RAG (Local)

本機 PDF RAG 示範：把香港衞生防護中心《季節性流感高峰期的家居衞生及消毒》PDF 切塊、用 Ollama embedding 寫入 Pinecone，再以本機 LLM 回答，並提供 Streamlit 介面。

## Architecture

1. **Ingest**：PDF → chunk → `nomic-embed-text` → Pinecone  
2. **Query**：問題 → 同款 embedding → Pinecone top-k → `qwen2.5:7b` 依引用段落回答  
3. **UI**：Streamlit（`app.py`）

刻意少用「整包 chain 黑盒」：流程拆在可讀的腳本裡，方便學習與監察延遲。

## Requirements

- Windows / macOS / Linux
- [Anaconda](https://www.anaconda.com/) 或 Python 3.10+
- [Ollama](https://ollama.com/) 並已拉取：
  - `nomic-embed-text`
  - `qwen2.5:7b`
- [Pinecone](https://www.pinecone.io/) API key  
  - 需一個 **768 維、cosine** 的一般向量 index（不要用帶 document schema、只能走 documents API 的 index）

## Setup

```bash
# 建議使用獨立 conda env（名稱可自訂）
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install pinecone python-dotenv pypdf langchain-text-splitters ollama streamlit
```

```bash
cp .env.example .env
# 編輯 .env，填入 PINECONE_API_KEY 與 index／namespace
```

下載 sample PDF（不進 Git，請自行放置於專案根目錄）：

- 檔名：`advice_household_disinfection_flu_chi.pdf`
- 來源：[CHP PDF](https://www.chp.gov.hk/files/pdf/advice_on_household_disinfection_at_the_height_of_influenza_season_chi.pdf)

## Usage

```bash
# Phase 1：抽出／切塊／寫入 Pinecone（含煙測 query）
python ingest.py

# 亦可分步查看
python extract_pdf.py
python chunk.py

# Phase 2：命令列問答
python ask.py "流感高峰期，家居表面應該用什麼漂白水比例消毒？"

# Phase 3：網頁介面
python -m streamlit run app.py
```

瀏覽器開啟 http://localhost:8501

### Ollama host 注意

若環境變數 `OLLAMA_HOST=0.0.0.0`（給 server 綁埠用），Python client 請連 `http://127.0.0.1:11434`（本專案腳本已這樣寫）。

## Project layout（建議進 Git 的檔案）

| File | 說明 |
|------|------|
| `extract_pdf.py` | 從 PDF 抽文字 |
| `chunk.py` | 切 chunk，方便人工檢查 |
| `ingest.py` | embed + upsert Pinecone + 煙測 |
| `ask.py` | 檢索 + LLM 回答（CLI） |
| `app.py` | Streamlit 介面 |
| `.env.example` | 環境變數範本 |
| `.gitignore` | 忽略秘密與本機產物 |
| `README.md` | 本說明 |

## 不要上傳

- `.env`（含 API key）
- `venv/`
- `*.pdf`、`extracted_text.txt`、`chunks.txt`
- 舊實驗檔如 `download.py`（另一份 PDF 的早期腳本；可選，預設不建議放進正式 repo）

## Metrics（System Analyst）

介面／CLI 會顯示：

- `embed_ms`：問題 embedding 延遲  
- `pinecone_ms`：向量查詢延遲  
- `llm_ms`：本機生成延遲  

另建議追蹤：檢索 score、無命中率、引用是否正確、錯誤率。

## License / 資料來源

PDF 內容來自香港衞生防護中心公開資料；本 repo 僅提供技術示範，不提供醫療建議。
