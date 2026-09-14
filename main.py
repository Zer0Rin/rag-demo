import os
import shutil

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./docs"

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    chunk_size=64,
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
)

prompt = PromptTemplate.from_template("""你是一个文档问答助手。请根据以下检索到的文档内容回答问题。
如果文档中没有相关信息，请直接说"文档中未找到相关内容"，不要编造答案。

检索到的文档内容：
{context}

问题：{question}

回答：""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def describe_source(doc) -> str:
    """格式化来源。只有 PDF（PyMuPDFLoader）才带 page 字段。

    DOCX 由 Docx2txtLoader 解析，没有分页概念：不能显示页码，
    否则会给出一个不存在的“第 1 页”，与“来源可追溯”相矛盾。
    """
    meta = doc.metadata or {}
    name = meta.get("file_name") or os.path.basename(str(meta.get("source", "未知文件")))
    page = meta.get("page")
    locator = f" · 第 {page + 1} 页" if isinstance(page, int) else " · 无分页信息"
    return f"📄 {name}{locator}\n{doc.page_content[:200]}..."


def ask(question):
    if not question.strip():
        return "", ""

    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    sources = "\n\n".join(describe_source(doc) for doc in docs)
    return answer, sources


def ingest_uploaded_file(file):
    if file is None:
        return "请先上传文件"

    from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    filename = os.path.basename(file.name)
    if filename.endswith(".pdf"):
        loader_class = PyMuPDFLoader
    elif filename.endswith((".docx", ".doc")):
        loader_class = Docx2txtLoader
    else:
        return f"不支持的文件类型：{filename}"

    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, filename)
    shutil.copy(file.name, dest)

    docs = loader_class(dest).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return f"⚠️ {filename} 未解析出任何文本（可能是扫描件或缺文字层），未入库"
    vectorstore.add_documents(chunks)

    return f"✅ 已入库：{filename}，共 {len(chunks)} 个片段"


def build_app():
    with gr.Blocks(title="本地知识库问答") as app:
        gr.Markdown("## 📚 本地知识库问答系统")

        with gr.Row():
            upload = gr.File(
                label="上传文档（PDF / DOCX）",
                file_types=[".pdf", ".docx", ".doc"],
            )
            upload_btn = gr.Button("入库", variant="secondary")
            upload_status = gr.Textbox(label="入库状态", interactive=False)
        upload_btn.click(ingest_uploaded_file, inputs=upload, outputs=upload_status)

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(label="你的问题", placeholder="输入问题后按 Enter...")
                btn = gr.Button("提问", variant="primary")
                answer = gr.Textbox(label="回答", lines=8)
            with gr.Column(scale=1):
                sources = gr.Textbox(label="参考来源", lines=12)
        btn.click(ask, inputs=question, outputs=[answer, sources])
        question.submit(ask, inputs=question, outputs=[answer, sources])
    return app


if __name__ == "__main__":
    build_app().launch()
