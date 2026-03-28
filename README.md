# 📚 Local RAG Q&A System

A local knowledge base Q&A system built with LangChain + ChromaDB + DeepSeek, supporting PDF and DOCX document ingestion and natural language querying.

## ✨ Features

- 📄 Support PDF and DOCX document upload and ingestion
- 🔍 Semantic retrieval based on vector similarity search
- 🤖 Answer generation powered by DeepSeek LLM
- 📌 Display source document and page number for each answer
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
git clone https://github.com/你的用户名/rag-demo.git
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

Place your PDF or DOCX files in the `docs/` folder, then run:
```bash
python ingest.py
```

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
5. Source document and page number are displayed alongside the answer