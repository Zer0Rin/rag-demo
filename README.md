# 📚 Local RAG Q&A System

A local knowledge base Q&A system built with LangChain + ChromaDB + DeepSeek, supporting PDF and DOCX document ingestion and natural language querying.

## ✨ Features

- 📄 Support PDF and DOCX document upload and ingestion
- 🔍 Semantic retrieval based on vector similarity search
- 🤖 Answer generation powered by DeepSeek LLM
- 📌 Display the source file for every answer (plus page number for PDFs)
- 🌐 Clean web interface built with Gradio

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | DeepSeek API (OpenAI-compatible) |
| Embedding | BAAI/bge-m3 via SiliconFlow |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Frontend | Gradio |
| Document Parsing | PyMuPDF, Docx2txt |

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/Zer0Rin/rag-demo.git
cd rag-demo
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure API keys**

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

**4. Ingest documents**

Two ways, both writing into the same local ChromaDB:

- **From the web UI (recommended):** start the app (step 5) and upload a document in the
  “上传文档” panel. `docs/` is created automatically and the number of ingested chunks is
  reported back.
- **From the CLI:** place your PDF or DOCX files in the `docs/` folder, then run:
```bash
mkdir -p docs
python ingest.py
```

Documents are split with `chunk_size=500` / `chunk_overlap=50`.

**5. Start the app**
```bash
python main.py
```

Open `http://localhost:7860` in your browser.

## 📁 Project Structure
```
rag-demo/
├── docs/          # Place your documents here
├── chroma_db/     # Vector database (auto-generated)
├── main.py        # Main app with Gradio UI
├── ingest.py      # Document ingestion script
├── .env           # API keys (not committed)
├── .env.example   # Environment variable template
└── requirements.txt
```

## ⚙️ How It Works

1. Documents are parsed and split into chunks
2. Each chunk is embedded using `BAAI/bge-m3` and stored in ChromaDB
3. On query, the top-4 most relevant chunks are retrieved
4. DeepSeek LLM generates an answer grounded in the retrieved context
5. The source file (and page number, for PDFs) is displayed alongside the answer

## ⚠️ Scope and limitations

This is a **prototype**, not a tuned retrieval system. Known gaps, stated on purpose:

- `chunk_size=500` / `chunk_overlap=50` / `top_k=4` are **starting values, not tuned
  optima** — no A/B comparison or retrieval evaluation has been run yet;
- no re-ranking, no hybrid (keyword + vector) retrieval, no query rewriting;
- answers are only as good as the retrieved chunks: there is no claim-level grounding
  check, so a wrong retrieval produces a confident wrong answer that still cites a real
  source;
- DOCX has no page concept, so only the file name is shown for `.docx`;
- documents with no extractable text layer (scanned PDFs) are rejected at ingest rather
  than silently indexed as empty;
- `langchain-community` is used only for the two document loaders and is being sunset
  upstream; migrating to `pymupdf` / `docx2txt` directly is the intended follow-up.

## 📊 Next steps

To turn this into something measurable: build a small gold set of (question, relevant
chunk) pairs, then report recall@k and answer-groundedness for a few chunk-size and
top-k settings instead of asserting that the current configuration is best.