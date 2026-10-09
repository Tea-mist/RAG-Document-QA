# 基于RAG的PDF文档智能问答系统

基于 **LangChain + Chroma向量数据库 + BGE中文嵌入模型 + DeepSeek大模型 + Gradio** 实现的本地PDF检索增强问答（RAG）系统。用户上传PDF文档后，系统自动解析并构建向量库，之后可针对文档内容进行多次提问，基于原文回答并附带片段溯源。

## ✨ 功能特性

1. **PDF文档解析**：支持上传PDF，自动完成文本提取与切片分块
2. **向量检索增强（RAG）**：将文档文本向量化存入Chroma，提问时检索最相关片段作为上下文
3. **一次上传、多次提问**：PDF仅在首次上传时解析构建向量库，后续修改问题直接问答，无需重复加载
4. **减少幻觉**：回答严格基于文档内容，文档中不存在的答案会明确提示，不编造
5. **时间感知**：自动读取系统当前年份注入提示词，支持年龄计算等时间相关推理
6. **中文优化**：使用BGE中文嵌入模型（bge-small-zh-v1.5），中文文档检索效果更佳
7. **国内镜像**：内置HuggingFace国内镜像，解决模型下载网络超时问题
8. **自动清理**：每次上传新PDF自动清除旧向量库，避免Chroma数据库冲突
9. **可视化界面**：基于Gradio的网页界面，操作简单、开箱即用

## 📦 技术栈

| 组件 | 说明 |
|------|------|
| Python | 开发语言 |
| LangChain | RAG应用框架 |
| Chroma | 轻量级本地向量数据库 |
| BGE嵌入模型 | sentence-transformers文本向量化 |
| DeepSeek API | 大模型问答推理 |
| PyPDF | PDF文档解析 |
| Gradio | Web可视化界面 |

## 📄 环境依赖

> Python 3.10 及以上版本

依赖清单（写入 `requirements.txt`）：

```txt
langchain
langchain-community
langchain-openai
langchain-text-splitters
langchain-core
chromadb
sentence-transformers
gradio
pypdf
## 🚀 环境部署

### 1. 创建虚拟环境

```
python -m venv .venv
```

### 2. 激活虚拟环境

```
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. 安装依赖

```
pip install -r requirements.txt
```

### 4. 配置 DeepSeek API 密钥

打开 `main.py`，将 `API_KEY` 替换为你自己的密钥：

```
API_KEY = "sk-你的DeepSeek密钥"
```

> 
> DeepSeek API 密钥需前往 DeepSeek 开放平台注册获取，并确保账户有余额。

### 5. 启动项目

```
python main.py
```

### 6. 访问系统

浏览器打开：`http://127.0.0.1:7860`

## 📖 使用方法

1. **上传 PDF**：点击网页中【上传 PDF 文档】区域，选择 PDF 文件，系统自动完成文档解析、分块、向量化并构建数据库
2. **输入问题**：在输入框输入你想问文档的问题
3. **开始问答**：点击【开始问答】按钮，返回 AI 回答及检索到的参考文档片段
4. **连续提问**：直接修改问题框内容，再次点击按钮即可继续提问，**无需重复上传 PDF**
5. **更换文档**：重新上传新的 PDF 即可，系统会自动重建向量库

## 📁 项目目录结构

```
RAG-Document-QA/
├── main.py                 # 主程序代码
├── requirements.txt        # 依赖包列表
├── README.md               # 项目说明文档
├── ./model/                # BGE嵌入模型缓存目录（首次运行自动下载）
├── ./chroma_db/            # Chroma向量数据库存储目录（自动生成）
└── *.pdf                   # 待测试的PDF文档
```

## ⚠️ 注意事项

1. 运行过程中**不要关闭终端窗口**，关闭则网页服务立即停止
2. 首次运行会自动下载 `BAAI/bge-small-zh-v1.5` 中文嵌入模型，需耐心等待下载完成
3. 需提前在 DeepSeek 开放平台获取 API Key 并确保账户有余额
4. 每次上传新 PDF，程序会自动删除旧向量库，避免数据库冲突
5. 建议提问文档内存在的内容，测试效果最佳

## 🧠 系统工作原理

```
用户上传PDF
    │
    ▼
① 文档加载：PyPDFLoader 读取PDF文本
    │
    ▼
② 文本分割：RecursiveCharacterTextSplitter 切分为文本块
    │
    ▼
③ 向量化存储：BGE嵌入模型 将文本转为向量，存入Chroma
    │
    ▼
用户提问
    │
    ▼
④ 相似度检索：检索与问题最相关的Top-K文档片段
    │
    ▼
⑤ 增强Prompt：检索片段 + 系统当前年份 + 用户问题 → 组装提示词
    │
    ▼
⑥ 生成回答：调用DeepSeek大模型，基于文档内容回答
    │
    ▼
输出结果：AI回答 + 参考文档片段溯源
```

**核心原理**：检索增强生成（Retrieval-Augmented Generation, RAG）—— 先从私有文档中检索相关内容作为上下文，再交给大模型生成回答，从而减少大模型幻觉，实现基于本地私有文档的智能问答。

## 📌 可扩展方向（可选）

- 支持 Word / TXT / Markdown 等更多文档格式
- 增加多轮对话记忆，支持上下文连续追问
- 增加相似度阈值过滤，剔除不相关的检索结果
- 支持多文档并行检索与批量导入
- 部署到云端服务器，支持多用户并发访问