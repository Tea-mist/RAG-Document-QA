import os
import datetime
import shutil
# 设置hf国内镜像，必须放在最前面
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import gradio as gr

# ========== 填入你的DeepSeek API Key ==========
API_KEY = "在此处填入你的DeepSeek API Key"
# ============================================

# 本地嵌入模型 BGE中文模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'trust_remote_code': True},
    cache_folder="./model",
    encode_kwargs={'normalize_embeddings': True}
)

# 调用DeepSeek大模型API
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.3
)

prompt_template = """
已知当前年份：{current_year}
基于下面的文档内容回答用户问题。如果文档里没有答案，直接说找不到相关内容，不要编造。
文档内容：{context}
用户问题：{question}
"""

# 上传PDF，只执行一次：加载、切片、构建向量库，返回retriever存入state
def load_pdf(pdf_file):
    if pdf_file is None:
        return None
    persist_path = "./chroma_db"
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)

    loader = PyPDFLoader(pdf_file.name)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    split_docs = splitter.split_documents(docs)
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_path)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

# 问答函数，只做检索+LLM推理，不再解析PDF
def predict(retriever, question):
    if retriever is None:
        return "⚠️ 请先上传PDF文档！"
    if not question.strip():
        return "⚠️ 请输入你的问题！"

    current_year = datetime.datetime.now().year
    source_docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in source_docs])

    final_prompt = prompt_template.format(
        current_year=current_year,
        context=context_text,
        question=question
    )
    ans = llm.invoke(final_prompt).content

    source_text = "\n\n📑 参考文档片段：\n"
    for idx, doc in enumerate(source_docs):
        source_text += f"\n【片段{idx+1}】{doc.page_content[:200]}..."
    return ans + source_text

# Gradio界面，增加state缓存检索器
with gr.Blocks(title="智能PDF问答机器人") as demo:
    gr.Markdown("# 🤖 基于RAG的本地文档智能问答系统")
    # 会话状态，用来保存retriever
    retriever_state = gr.State(value=None)

    file = gr.File(label="上传PDF文档")
    q = gr.Textbox(label="在这里输入你的问题")
    a = gr.Textbox(label="AI回答结果", lines=12)
    btn = gr.Button("开始问答")

    # 上传PDF触发：加载文档，存入state
    file.upload(fn=load_pdf, inputs=[file], outputs=[retriever_state])
    # 点击问答：使用state里缓存好的retriever，直接问答
    btn.click(fn=predict, inputs=[retriever_state, q], outputs=[a])

if __name__ == "__main__":
    demo.launch()
