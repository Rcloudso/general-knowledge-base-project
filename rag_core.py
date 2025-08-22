import os
import time
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache
from loguru import logger
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document
import datetime
from dotenv import load_dotenv
from db_manager import DatabaseManager


# 日志处理
logger.remove()
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

current_date = datetime.datetime.now().strftime("%Y-%m-%d")

main_log_filename = log_dir / f"{current_date}.log"

logger.add(
    main_log_filename,
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


# 动态日志级别处理类
class LevelFileSink:
    def __init__(self, log_dir, level):
        self.log_dir = Path(log_dir)
        self.level = level

    def write(self, message):
        record = message.record
        if record["level"].name == self.level:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            level_log_filename = self.log_dir / f"{timestamp}_{self.level.lower()}.log"

            with open(level_log_filename, "a", encoding="utf-8") as f:
                f.write(
                    f"{record['time'].strftime('%Y-%m-%d %H:%M:%S')} | {record['level'].name} | {record['message']}\n"
                )


# 为关键级别添加动态日志文件
critical_levels = ["ERROR", "WARNING", "CRITICAL"]

for level in critical_levels:
    sink = LevelFileSink(log_dir, level)
    logger.add(
        sink.write,
        filter=lambda record, level=level: record["level"].name == level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

logger.add(
    lambda msg: print(msg),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

load_dotenv()

# 获取环境变量，提供默认值
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
)
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "z-ai/glm-4.5-air:free")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
SESSIONS_DB_PATH = os.getenv("SESSIONS_DB_PATH", "./db/sessions.db")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


# 使用lru_cache缓存embedding模型，避免重复加载
@lru_cache(maxsize=1)
def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """获取并缓存embedding模型"""
    logger.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


class RAGCore:
    def __init__(self, directory_path: Optional[str] = None):
        """初始化RAG核心组件

        Args:
            directory_path: 文档目录路径，如果提供则创建新的向量数据库
        """
        self.directory_path = directory_path
        self.db_path = CHROMA_DB_PATH
        self.hash_file_path = Path(self.db_path) / "file_hashes.json"
        self.embeddings = get_embeddings()

        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found in environment variables")

        self.llm = ChatOpenAI(
            base_url=OPENAI_BASE_URL,
            openai_api_key=OPENAI_API_KEY,
            model_name=OPENAI_MODEL_NAME,
        )
        self.db = None
        self.qa_chain = None
        self.db_manager = DatabaseManager(SESSIONS_DB_PATH)
        self.current_session_id = None

        if directory_path:
            self.create_database()

    def _load_documents_by_type(
        self, glob_pattern: str, loader_cls: Any, loader_kwargs: Dict[str, Any] = None
    ) -> List[Document]:
        """加载特定类型的文档

        Args:
            glob_pattern: 文件匹配模式
            loader_cls: 加载器类
            loader_kwargs: 加载器参数

        Returns:
            加载的文档列表
        """
        loader_kwargs = loader_kwargs or {}
        try:
            loader = DirectoryLoader(
                self.directory_path,
                glob=glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                show_progress=True,
                use_multithreading=True,
            )
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents matching {glob_pattern}")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents matching {glob_pattern}: {e}")
            return []

    def _load_txt_documents(self) -> List[Document]:
        """加载TXT文档，尝试多种编码"""
        encodings = ["utf-8", "gbk", "gb2312", "gb18030", "big5"]
        all_docs = []

        for encoding in encodings:
            if all_docs:
                break

            docs = self._load_documents_by_type(
                glob_pattern="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": encoding},
            )

            if docs:
                logger.info(
                    f"Successfully loaded TXT documents with {encoding} encoding"
                )
                all_docs.extend(docs)

        return all_docs

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的SHA256哈希值"""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _load_file_hashes(self) -> Dict[str, Dict[str, Any]]:
        if not self.hash_file_path.exists():
            return {}
        try:
            with open(self.hash_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading file hashes: {e}")
            return {}

    def _save_file_hashes(self, file_hashes: Dict[str, Dict[str, Any]]) -> None:
        """保存文件哈希信息"""
        try:
            self.hash_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.hash_file_path, "w", encoding="utf-8") as f:
                json.dump(file_hashes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving file hashes: {e}")

    def _normalize_path(self, file_path: str) -> str:
        """规范化文件路径，确保路径格式统一

        Args:
            file_path: 原始文件路径

        Returns:
            规范化后的文件路径
        """
        if file_path.startswith("./") or file_path.startswith(".\\"):
            file_path = file_path[2:]

        abs_path = os.path.abspath(file_path)
        cwd = os.path.abspath(os.getcwd())
        if abs_path.startswith(cwd):
            rel_path = os.path.relpath(abs_path, cwd)
            return rel_path.replace("\\", "/")
        else:
            return file_path.replace("\\", "/")

    def _get_modified_files(self, file_hashes: Dict[str, Dict[str, Any]]) -> List[str]:
        """获取新增或修改的文件列表"""
        if not self.directory_path:
            return []
        modified_files = []
        supported_extensions = {".pdf", ".txt", ".doc", ".docx"}
        current_files = set()

        normalized_to_original = {}
        normalized_hashes = {}

        for original_path, hash_info in file_hashes.items():
            norm_path = self._normalize_path(original_path)
            normalized_hashes[norm_path] = hash_info
            normalized_to_original[norm_path] = original_path

        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                if file_ext in supported_extensions:
                    norm_file_path = self._normalize_path(file_path)
                    current_files.add(norm_file_path)
                    current_hash = self._calculate_file_hash(file_path)
                    current_mtime = os.path.getmtime(file_path)

                    if norm_file_path not in normalized_hashes:
                        modified_files.append(file_path)
                        logger.info(f"New file detected: {file_path}")
                    else:
                        stored_hash = normalized_hashes[norm_file_path].get("hash", "")
                        stored_mtime = normalized_hashes[norm_file_path].get("mtime", 0)
                        if current_hash != stored_hash or current_mtime != stored_mtime:
                            modified_files.append(file_path)
                            logger.info(f"Modified file detected: {file_path}")

        deleted_files = []
        for norm_path in normalized_hashes.keys():
            original_path = normalized_to_original.get(norm_path, norm_path)
            if (
                norm_path not in current_files
                and os.path.exists(original_path) is False
            ):
                deleted_files.append(original_path)
                logger.info(f"Deleted file detected: {original_path}")

        if deleted_files:
            for deleted_file in deleted_files:
                norm_deleted = self._normalize_path(deleted_file)
                for original_path in list(file_hashes.keys()):
                    if self._normalize_path(original_path) == norm_deleted:
                        del file_hashes[original_path]

            self._save_file_hashes(file_hashes)
            logger.info(
                f"Cleaned up {len(deleted_files)} deleted files from hash records"
            )

        return modified_files

    def load_documents(self) -> List[Document]:
        """从指定目录加载所有支持的文档类型（支持增量更新）"""
        if not self.directory_path:
            logger.error("No directory path specified")
            return []

        if not os.path.exists(self.directory_path):
            logger.error(f"Directory does not exist: {self.directory_path}")
            return []

        logger.info(f"Loading documents from {self.directory_path}")

        file_hashes = self._load_file_hashes()
        modified_files = self._get_modified_files(file_hashes)

        if not modified_files:
            logger.info("No new or modified files detected")
            return []

        documents = []
        for file_path in modified_files:
            file_ext = Path(file_path).suffix.lower()
            try:
                if file_ext == ".pdf":
                    docs = PyPDFLoader(file_path).load()
                elif file_ext == ".txt":
                    docs = None
                    for encoding in ["utf-8", "gbk", "gb2312", "gb18030", "big5"]:
                        try:
                            docs = TextLoader(file_path, encoding=encoding).load()
                            break
                        except UnicodeDecodeError:
                            continue
                    if not docs:
                        continue
                elif file_ext in [".doc", ".docx"]:
                    docs = UnstructuredWordDocumentLoader(file_path).load()
                else:
                    continue
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")

            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

        logger.info(
            f"Total documents loaded: {len(documents)} (from {len(modified_files)} modified files)"
        )
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成更小的块

        Args:
            documents: 要分割的文档列表

        Returns:
            分割后的文档块列表
        """
        logger.info("Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(texts)} chunks")
        return texts

    def create_database(self) -> bool:
        """从文档创建Chroma向量数据库（支持增量更新）

        Returns:
            是否成功创建数据库
        """
        logger.info("Loading documents...")
        documents = self.load_documents()

        self.clean_file_hashes()

        file_hashes = self._load_file_hashes()

        if not documents:
            if os.path.exists(self.db_path):
                logger.info("Database already exists and no new files detected")
                return self.load_database()
            else:
                logger.warning("No documents found in the specified directory.")
                return False

        logger.info("Splitting documents...")
        texts = self.split_documents(documents)

        logger.info("Creating/updating Chroma vector database...")
        try:
            # 更新文件哈希信息
            for file_path in self._get_modified_files(file_hashes):
                if os.path.exists(file_path):
                    # 使用规范化的路径作为键
                    norm_path = self._normalize_path(file_path)
                    file_hashes[norm_path] = {
                        "hash": self._calculate_file_hash(file_path),
                        "mtime": os.path.getmtime(file_path),
                    }
            self._save_file_hashes(file_hashes)
            if os.path.exists(self.db_path):
                existing_db = Chroma(
                    persist_directory=self.db_path, embedding_function=self.embeddings
                )
                existing_db.add_documents(texts)
                self.db = existing_db
                logger.info(f"Database updated successfully at {self.db_path}")
            else:
                self.db = Chroma.from_documents(
                    texts, self.embeddings, persist_directory=self.db_path
                )
                logger.info(f"Database created successfully at {self.db_path}")

            self.setup_qa_chain()
            return True
        except Exception as e:
            logger.error(f"Error creating/updating database: {e}")
            return False

    def clean_file_hashes(self) -> bool:
        """清理文件哈希记录，移除重复的文件记录

        Returns:
            是否成功清理文件哈希记录
        """
        try:
            file_hashes = self._load_file_hashes()
            if not file_hashes:
                logger.info("No file hash records to clean")
                return True

            normalized_paths = {}
            cleaned_hashes = {}

            # 遍历所有文件哈希记录
            for original_path, hash_info in file_hashes.items():
                norm_path = self._normalize_path(original_path)
                if norm_path in normalized_paths:
                    # 比较修改时间，保留最新的记录
                    existing_mtime = cleaned_hashes[normalized_paths[norm_path]].get(
                        "mtime", 0
                    )
                    current_mtime = hash_info.get("mtime", 0)
                    if current_mtime > existing_mtime:
                        cleaned_hashes[norm_path] = hash_info
                        normalized_paths[norm_path] = norm_path
                        logger.info(f"Replaced duplicate record for {norm_path}")
                else:
                    cleaned_hashes[norm_path] = hash_info
                    normalized_paths[norm_path] = norm_path

            self._save_file_hashes(cleaned_hashes)
            logger.info(
                f"Cleaned file hash records: {len(file_hashes)} -> {len(cleaned_hashes)}"
            )
            return True
        except Exception as e:
            logger.error(f"Error cleaning file hash records: {e}")
            return False

    def load_database(self, force_recreate: bool = False) -> bool:
        """加载现有的Chroma向量数据库

        Args:
            force_recreate: 是否强制重新创建数据库，用于解决维度不匹配问题

        Returns:
            是否成功加载数据库
        """
        logger.info(f"Loading existing Chroma vector database from {self.db_path}...")

        # 如果强制重新创建或数据库不存在，则创建新数据库
        if force_recreate:
            logger.warning(
                "Force recreating database due to user request or dimension mismatch"
            )
            if os.path.exists(self.db_path) and self.directory_path:
                import shutil

                try:
                    # 备份原数据库
                    backup_path = f"{self.db_path}_backup_{int(time.time())}"
                    shutil.copytree(self.db_path, backup_path)
                    logger.info(f"Backed up existing database to {backup_path}")
                    shutil.rmtree(self.db_path)
                    logger.info(f"Removed existing database at {self.db_path}")
                except Exception as e:
                    logger.error(f"Error backing up/removing database: {e}")

            if self.directory_path:
                return self.create_database()
            else:
                logger.error(
                    "Cannot recreate database: No document directory specified"
                )
                return False

        if os.path.exists(self.db_path):
            try:
                self.db = Chroma(
                    persist_directory=self.db_path, embedding_function=self.embeddings
                )
                logger.info("Database loaded successfully.")
                self.setup_qa_chain()
                return True
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                if "dimension" in str(e).lower() and self.directory_path:
                    logger.warning(
                        "Embedding dimension mismatch detected. Try running with --force-recreate"
                    )
                return False
        logger.warning(f"Database path does not exist: {self.db_path}")
        return False

    def setup_qa_chain(self) -> None:
        """设置问答链用于回答问题"""
        logger.info("Setting up QA chain...")
        if self.db:
            retriever = self.db.as_retriever(
                # 高精度相似度检索
                # search_type="similarity_score_threshold",
                # search_kwargs={"k": 6, "score_threshold": 0.7},
                # 最大边际相关性检索
                search_type="mmr",
                search_kwargs={"k": 4, "lambda_mult": 0.5},
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            logger.info("QA chain setup complete")

    def query(self, question: str) -> tuple[str, List[Document]]:
        """使用问答链回答问题

        Args:
            question: 要回答的问题

        Returns:
            包含回答和源文档的元组
        """
        if not self.qa_chain:
            logger.error("QA chain not set up. Please load or create a database first.")
            return "QA chain is not set up. Please create or load a database first.", []

        logger.info(f"Processing question: {question}")
        try:
            result = self.qa_chain.invoke({"query": question})
            logger.info("Question answered successfully")
            return result["result"], result["source_documents"]
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing question: {str(e)}", []

    def create_session(self, session_id=None):
        """创建新的会话

        Args:
            session_id: 会话ID，如果不提供则自动生成

        Returns:
            会话ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        system_message = """你是一个有用的AI助手，请根据会话历史和参考资料回答问题。

遵循以下规则：
1. 优先考虑会话历史中的信息，这代表了用户与你之间的交互记录。
2. 仔细分析参考资料中的内容，特别关注与当前问题最相关的部分。
3. 参考资料按相关性排序，前面的文档通常更相关。
4. 如果参考资料中包含与问题直接相关的信息，请详细解答并引用来源。
5. 如果会话历史和参考资料中都没有相关信息，请如实告知用户。
6. 回答要简洁明了，直接回应用户的问题。
7. 不要编造信息，只使用会话历史和提供的参考资料。"""

        self.db_manager.create_session(session_id, system_message)
        self.current_session_id = session_id
        logger.info(f"Created new session: {session_id}")
        return session_id

    def switch_session(self, session_id):
        """切换到指定会话

        Args:
            session_id: 要切换到的会话ID

        Returns:
            是否成功切换
        """
        if self.db_manager.session_exists(session_id):
            self.current_session_id = session_id
            logger.info(f"Switched to session: {session_id}")
            return True
        else:
            logger.warning(f"Session {session_id} not found")
            return False

    def add_message(self, role, content):
        """向当前会话添加消息

        Args:
            role: 消息角色 ("user", "assistant", "system")
            content: 消息内容

        Returns:
            是否成功添加
        """
        if not self.current_session_id:
            self.create_session()

        return self.db_manager.add_message(self.current_session_id, role, content)

    def get_session_messages(self, session_id=None):
        """获取指定会话的所有消息

        Args:
            session_id: 会话ID，如果不提供则使用当前会话

        Returns:
            会话消息列表
        """
        session_id = session_id or self.current_session_id
        if not session_id:
            return []
        return self.db_manager.get_session_messages(session_id)

    def list_sessions(self):
        """列出所有会话

        Returns:
            会话ID和创建时间的列表
        """
        return self.db_manager.list_sessions()

    def delete_session(self, session_id):
        """删除指定会话

        Args:
            session_id: 要删除的会话ID

        Returns:
            是否成功删除
        """
        result = self.db_manager.delete_session(session_id)

        if result and self.current_session_id == session_id:
            self.current_session_id = None

        if result:
            logger.info(f"Deleted session: {session_id}")
        else:
            logger.warning(f"Session {session_id} not found for deletion")

        return result

    def query_stream(self, question: str, session_id=None):
        """使用问答链回答问题，支持流式输出，并保存会话上下文

        Args:
            question: 要回答的问题
            session_id: 会话ID，如果不提供则使用当前会话

        Returns:
            生成器，产生部分回答和最终的源文档
        """
        if not self.qa_chain:
            logger.error("QA chain not set up. Please load or create a database first.")
            yield "QA chain is not set up. Please create or load a database first.", []
            return

        if session_id:
            if not self.db_manager.session_exists(session_id):
                self.create_session(session_id)
            else:
                self.current_session_id = session_id
        elif not self.current_session_id:
            self.create_session()

        self.add_message("user", question)

        logger.info(f"Processing question with streaming: {question}")
        try:
            retriever = self.db.as_retriever(search_kwargs={"k": 6})
            docs = retriever.invoke(question)

            if hasattr(docs[0], "metadata") and "score" in docs[0].metadata:
                docs = sorted(
                    docs, key=lambda x: x.metadata.get("score", 0), reverse=True
                )

            context_parts = []
            for i, doc in enumerate(docs):
                source = (
                    doc.metadata.get("source", "未知来源")
                    if hasattr(doc, "metadata")
                    else "未知来源"
                )
                context_parts.append(
                    f"文档{i+1}（来源：{source}）:\n{doc.page_content}"
                )
            context = "\n\n".join(context_parts)

            messages = self.get_session_messages()

            if messages and messages[-1]["role"] == "user":
                messages.pop()

            messages.append(
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n参考资料（按相关性排序）：\n{context}",
                }
            )

            if len(docs) > 0:
                logger.info(f"Retrieved {len(docs)} documents for question: {question}")
            else:
                logger.warning(f"No documents retrieved for question: {question}")

            response = ""
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    response += chunk.content
                    yield response, docs

            self.add_message("assistant", response)
            logger.info("Streaming question answered successfully")
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            error_msg = f"Error processing question: {str(e)}"
            self.add_message("assistant", error_msg)
            yield error_msg, []


def main():
    """主函数，处理命令行参数并运行RAG系统"""
    import argparse

    parser = argparse.ArgumentParser(description="运行基于检索增强生成的问答系统")
    parser.add_argument("--directory_path", type=str, help="文档目录路径")
    parser.add_argument("--db_path", type=str, help="数据库存储路径")
    parser.add_argument("--create_db", action="store_true", help="强制创建新数据库")
    parser.add_argument(
        "--force_recreate",
        action="store_true",
        help="强制重新创建数据库，用于解决维度不匹配问题",
    )
    parser.add_argument("--query", type=str, help="单次查询模式，提供问题后退出")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")

    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add("rag_core.log", rotation="10 MB", level="DEBUG")
        logger.add(lambda msg: print(msg), level="DEBUG")
        logger.debug("Debug logging enabled")

    # 创建RAG实例
    rag_kwargs = {}
    if args.directory_path:
        rag_kwargs["directory_path"] = args.directory_path
    if args.db_path:
        os.environ["CHROMA_DB_PATH"] = args.db_path

    rag = RAGCore(**rag_kwargs)

    if args.create_db or not rag.load_database(force_recreate=args.force_recreate):
        if not args.force_recreate:
            logger.info("Creating new database...")
            if not rag.create_database():
                logger.error("Failed to create database. Exiting.")
                return

    # 单次查询模式
    if args.query:
        answer, sources = rag.query(args.query)
        print(f"\n问题: {args.query}")
        print(f"回答: {answer}")

        print("\n来源:")
        for i, doc in enumerate(sources):
            print(f"来源 {i+1}:\n{doc.page_content}\n")
        return

    # 交互式模式
    print("\n欢迎使用RAG问答系统！输入'exit'退出。")
    while True:
        try:
            question = input("\n请输入您的问题: ")
            if question.lower() in ["exit", "quit", "q", "退出"]:
                break

            answer, sources = rag.query(question)
            print(f"\n回答: {answer}")

            print("\n来源:")
            for i, doc in enumerate(sources):
                print(f"来源 {i+1}:\n{doc.page_content}\n")
        except KeyboardInterrupt:
            print("\n程序被用户中断，正在退出...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"发生错误: {e}")

    print("感谢使用，再见！")


if __name__ == "__main__":
    main()
