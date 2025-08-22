# RAG知识库问答系统

一个基于检索增强生成（Retrieval-Augmented Generation, RAG）技术的智能知识库问答系统，能够从本地文档中提取信息并提供准确的问答服务。已简单自测四本古籍的问答功能，通过streamlit启动web应用尚存在回答不准的问题。

## ✨ 核心功能

### 📄 多格式文档支持
- 支持 PDF、TXT、DOC/DOCX 等多种文档格式
- 自动处理中文编码（UTF-8、GBK、GB2312等）

### 🔍 智能检索能力
- 基于向量数据库的语义搜索
- 支持相似度检索和关键词匹配
- 自动文档分块和向量化处理

### 🤖 AI问答生成
- 集成 OpenAI API 进行智能问答
- 支持上下文感知的多轮对话
- 实时流式输出响应

### ⚙️ 配置管理
- 环境变量配置系统
- 模型缓存机制，避免重复下载
- 灵活的数据库管理

### 📊 日志与监控
- 使用 loguru 进行高级日志管理
- 彩色控制台输出
- 日志文件轮转和归档

### 💬 会话管理
- 完整的对话历史保存
- 上下文感知的问答生成
- 支持会话恢复和导出（后续迭代）

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 支持CUDA的GPU（可选，用于加速处理）

### 安装依赖

```bash
# 创建环境
conda create -n rag_env python=3.10
conda activate rag_env
```
或者

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows


# 克隆项目
git clone https://github.com/Rcloudso/general-knowledge-base-project
cd general-knowledge-base-project

# 安装依赖
pip install -r requirements.txt
```

## ⚙️ 配置说明

### 🔧 环境变量配置

在运行程序前，请设置以下环境变量：

#### 方法一：命令行设置
```bash
# Linux/Mac
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="openai/gpt-3.5-turbo"
export EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
export CHROMA_DB_PATH="./chroma_db"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-api-key"
$env:OPENAI_BASE_URL="https://openrouter.ai/api/v1"
$env:OPENAI_MODEL="openai/gpt-3.5-turbo"
$env:EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
$env:CHROMA_DB_PATH="./chroma_db"

# Windows (CMD)
set OPENAI_API_KEY=your-openai-api-key
set OPENAI_BASE_URL=https://openrouter.ai/api/v1
set OPENAI_MODEL=openai/gpt-3.5-turbo
set EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
set CHROMA_DB_PATH=./chroma_db
```

#### 方法二：创建 `.env` 文件
```env
# OpenRouter API配置
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-3.5-turbo

# 嵌入模型配置
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 数据库配置
CHROMA_DB_PATH=./chroma_db

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# 可选：代理设置（如果需要）
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=http://proxy.example.com:8080
```

#### 获取API密钥
1. 访问 [OpenAI官网](https://openai.com/) 或者任意支持OpenAI API的模型提供商注册账号
2. 在控制台获取API密钥
3. 将密钥设置到环境变量或 `.env` 文件中

### 支持的模型
- **嵌入模型**: BAAI/bge-small-zh-v1.5, BAAI/bge-large-zh-v1.5, 其他HuggingFace模型
- **LLM模型**: 所有OpenAI支持的模型（GPT-3.5, GPT-4, Claude, Gemini等）

## 🎯 使用方法

### 基本使用

1. **启动应用**:
   ```bash
   python rag_core.py
   ```

2. **命令行交互**:
   - 输入问题或指令进行对话
   - 输入 `/help` 查看可用命令
   - 输入 `/exit` 退出程序

### 命令行参数

```bash
# 指定文档目录
python rag_core.py --directory_path ./documents

# 指定数据库路径
python rag_core.py --db_path ./my_chroma_db

# 强制创建新数据库
python rag_core.py --create_db

# 强制重新创建数据库
python rag_core.py --force_recreate

# 单次查询模式
python rag_core.py --query "你的问题"

# 启用调试日志
python rag_core.py --debug

# 查看帮助
python rag_core.py --help
```

### 交互命令
- `/help` - 显示帮助信息
- `/exit` - 退出程序

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

## 📁 项目结构

```
General Knowledge Base/
├── 📂 data/                  # 知识库文档目录
├── 📂 logs/                  # 日志目录（自动生成）
├── 📂 tests/                 # 测试文件目录
│   ├── 📄 __init__.py
│   └── 📄 test_db.py        # 数据库管理器测试
├── 📂 db/                   # 数据库目录（自动生成）
│   ├── 📄 sessions.db       # 会话数据库
│   └── 📂 chroma_db/        # 向量数据库（自动生成）
├── 📄 app.py                 # Streamlit Web应用
├── 📄 rag_core.py           # RAG核心模块
├── 📄 db_manager.py         # 数据库管理器
├── 📄 requirements.txt      # 项目依赖
├── 📄 .env.example          # 环境变量示例
├── 📄 README.md             # 项目说明
└── 📄 .gitignore           # Git忽略文件
```
## 注意事项

- 首次运行时需要下载嵌入模型，可能需要一些时间
- 对于大型文档集合，创建数据库可能需要较长时间
- 确保文档目录中包含有效的文档文件（PDF、TXT、DOC/DOCX）

## 📞 联系方式

- **作者**: Rcloudso & AI Team
- **邮箱**: 1123025945@qq.com
- **GitHub**: [https://github.com/Rcloudso](https://github.com/Rcloudso)
- **问题反馈**: 请在 GitHub Issues 中提交问题或建议

## 🤝 致谢

感谢以下开源项目和技术：
- [OpenAI](https://openai.com/) - 提供多模型LLM API服务
- [ChromaDB](https://www.trychroma.com/) - 开源向量数据库
- [HuggingFace](https://huggingface.co/) - 提供优秀的嵌入模型
- [LangChain](https://www.langchain.com/) - LLM应用开发框架

---

⭐ 如果这个项目对您有帮助，请给个星标支持！