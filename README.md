# RAG知识库问答系统

这是一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的知识库问答系统，可以从本地文档中提取信息并回答用户问题。

## 功能特点

- 支持多种文档格式（PDF、TXT、DOC/DOCX）
- 自动处理中文编码问题（支持UTF-8、GBK、GB2312等）
- 使用向量数据库存储文档内容，支持语义搜索
- 使用OpenRouter API进行问答生成
- 支持环境变量配置API密钥和其他参数
- 缓存嵌入模型，避免重复下载
- 使用loguru进行高级日志管理，支持彩色输出和文件轮转
- 支持流式输出，实时显示生成的回答
- 支持会话管理，保存对话历史并将上下文提供给LLM

## 安装依赖

```bash
pip install langchain langchain_community langchain_openai chromadb sentence_transformers python-dotenv unstructured pypdf python-docx loguru
```

## 配置

在项目根目录创建`.env`文件，配置以下环境变量：

```
OPENROUTER_API_KEY=your_openrouter_api_key
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
OPENROUTER_MODEL_NAME=z-ai/glm-4.5-air:free
CHROMA_DB_PATH=./chroma_db
```

## 使用方法

### 命令行参数

```
python rag_core.py --help
```

参数说明：
- `--directory_path`: 文档目录路径
- `--db_path`: 数据库存储路径
- `--create_db`: 强制创建新数据库
- `--force_recreate`: 强制重新创建数据库，用于解决维度不匹配问题
- `--query`: 单次查询模式，提供问题后退出
- `--debug`: 启用调试日志

### 示例

1. 创建新的知识库：

```bash
python rag_core.py --directory_path ./documents --create_db
```

2. 使用现有知识库进行交互式问答：

```bash
python rag_core.py
```

3. 单次查询模式：

```bash
python rag_core.py --query "什么是八字命理？"
```

## 代码结构

- `rag_core.py`: 主程序文件，包含RAG系统的核心实现
- `.env`: 环境变量配置文件
- `chroma_db/`: 默认的向量数据库存储目录

## 注意事项

- 首次运行时需要下载嵌入模型，可能需要一些时间
- 对于大型文档集合，创建数据库可能需要较长时间
- 确保文档目录中包含有效的文档文件（PDF、TXT、DOC/DOCX）