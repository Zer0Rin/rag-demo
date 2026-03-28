import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"

def ingest():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())
        else:
            continue
        print(f"已加载: {filename}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"切块完成，共 {len(chunks)} 个 chunk")

    load_dotenv()

    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        chunk_size=64
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("向量库构建完成，已持久化到", CHROMA_DIR)

if __name__ == "__main__":
    ingest()