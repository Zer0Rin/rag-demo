import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

load_dotenv()

CHROMA_DIR = "./chroma_db"

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    chunk_size=64
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3
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


def ask(question):
    if not question.strip():
        return "", ""

    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = "\n\n".join([
        f"📄 {doc.metadata.get('file_name', doc.metadata.get('source', '未知文件'))} · 第 {(doc.metadata.get('page') or 0) + 1} 页\n{doc.page_content[:200]}..."
        for doc in docs
    ])
    return answer, sources


def ingest_uploaded_file(file):
    if file is None:
        return "请先上传文件"

    import shutil
    from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    filename = os.path.basename(file.name)
    dest = os.path.join("./docs", filename)
    shutil.copy(file.name, dest)

    if filename.endswith(".pdf"):
        loader = PyMuPDFLoader(dest)
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        loader = Docx2txtLoader(dest)
    else:
        return f"不支持的文件类型：{filename}"

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)

    return f"✅ 已入库：{filename}，共 {len(chunks)} 个片段"


with gr.Blocks(title="本地知识库问答") as app:
    gr.Markdown("## 📚 本地知识库问答系统")

    with gr.Row():
        upload = gr.File(label="上传文档（PDF / DOCX）", file_types=[".pdf", ".docx", ".doc"])
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

app.launch()