import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

from loguru import logger
from db_manager import DatabaseManager  # 导入数据库管理器

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv

# 配置日志
logger.remove()  # 移除默认处理器
logger.add(
    "rag_core.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)  # 添加文件日志
logger.add(
    lambda msg: print(msg),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)  # 添加控制台日志

# 加载环境变量
load_dotenv()

# 获取环境变量，提供默认值
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# 使用与原数据库相同的嵌入模型，确保维度匹配
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
)
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "z-ai/glm-4.5-air:free")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
SESSIONS_DB_PATH = os.getenv("SESSIONS_DB_PATH", "./sessions.db")


# 使用lru_cache缓存embedding模型，避免重复加载
@lru_cache(maxsize=1)
def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """获取并缓存embedding模型"""
    logger.info(f"Loading embedding model: {model_name}")
    # 确保使用与原来相同维度的模型
    return HuggingFaceEmbeddings(model_name=model_name)


class RAGCore:
    def __init__(self, directory_path: Optional[str] = None):
        """初始化RAG核心组件

        Args:
            directory_path: 文档目录路径，如果提供则创建新的向量数据库
        """
        self.directory_path = directory_path
        self.db_path = CHROMA_DB_PATH

        # 使用缓存获取embeddings模型
        self.embeddings = get_embeddings()

        # 检查API密钥是否存在
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not found in environment variables")

        # 使用ChatOpenAI类初始化LLM，从环境变量获取配置
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=OPENROUTER_API_KEY,
            model_name=OPENROUTER_MODEL_NAME,
            default_headers={"HTTP-Referer": "https://localhost:3000"},
        )

        self.db = None
        self.qa_chain = None
        
        # 初始化数据库管理器
        self.db_manager = DatabaseManager(SESSIONS_DB_PATH)
        self.current_session_id = None

        # 如果提供了目录路径，则创建数据库
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
            if all_docs:  # 如果已经成功加载了文档，就不再尝试其他编码
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

    def load_documents(self) -> List[Document]:
        """从指定目录加载所有支持的文档类型"""
        if not self.directory_path:
            logger.error("No directory path specified")
            return []

        # 确保目录存在
        if not os.path.exists(self.directory_path):
            logger.error(f"Directory does not exist: {self.directory_path}")
            return []

        logger.info(f"Loading documents from {self.directory_path}")
        documents = []

        # 加载PDF文件
        pdf_docs = self._load_documents_by_type("**/*.pdf", PyPDFLoader)
        documents.extend(pdf_docs)

        # 加载TXT文件（尝试多种编码）
        txt_docs = self._load_txt_documents()
        documents.extend(txt_docs)

        # 加载DOC文件
        doc_docs = self._load_documents_by_type(
            "**/*.doc", UnstructuredWordDocumentLoader
        )
        documents.extend(doc_docs)

        # 加载DOCX文件
        docx_docs = self._load_documents_by_type(
            "**/*.docx", UnstructuredWordDocumentLoader
        )
        documents.extend(docx_docs)

        logger.info(f"Total documents loaded: {len(documents)}")
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
        """从文档创建Chroma向量数据库

        Returns:
            是否成功创建数据库
        """
        logger.info("Loading documents...")
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found in the specified directory.")
            return False

        logger.info("Splitting documents...")
        texts = self.split_documents(documents)

        logger.info("Creating Chroma vector database...")
        try:
            self.db = Chroma.from_documents(
                texts, self.embeddings, persist_directory=self.db_path
            )
            self.db.persist()
            logger.info(f"Database created successfully at {self.db_path}")
            self.setup_qa_chain()
            return True
        except Exception as e:
            logger.error(f"Error creating database: {e}")
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
                    # 删除原数据库
                    shutil.rmtree(self.db_path)
                    logger.info(f"Removed existing database at {self.db_path}")
                except Exception as e:
                    logger.error(f"Error backing up/removing database: {e}")

            # 如果有文档目录，创建新数据库
            if self.directory_path:
                return self.create_database()
            else:
                logger.error(
                    "Cannot recreate database: No document directory specified"
                )
                return False

        # 正常加载数据库
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
                # 如果是维度不匹配错误且有文档目录，建议重新创建
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
            # 使用与query_stream方法相同的检索参数
            retriever = self.db.as_retriever(search_kwargs={"k": 6})
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
            # 使用invoke方法替代直接调用，解决Chain.__call__弃用警告
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
        
        # 系统提示消息
        system_message = """你是一个有用的AI助手，请根据会话历史和参考资料回答问题。

遵循以下规则：
1. 优先考虑会话历史中的信息，这代表了用户与你之间的交互记录。
2. 仔细分析参考资料中的内容，特别关注与当前问题最相关的部分。
3. 参考资料按相关性排序，前面的文档通常更相关。
4. 如果参考资料中包含与问题直接相关的信息，请详细解答并引用来源。
5. 如果会话历史和参考资料中都没有相关信息，请如实告知用户。
6. 回答要简洁明了，直接回应用户的问题。
7. 不要编造信息，只使用会话历史和提供的参考资料。"""
        
        # 使用数据库管理器创建会话
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
        
        # 如果删除的是当前会话，则重置当前会话ID
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
            
        # 确保有活跃会话
        if session_id:
            if not self.db_manager.session_exists(session_id):
                self.create_session(session_id)
            else:
                self.current_session_id = session_id
        elif not self.current_session_id:
            self.create_session()
            
        # 添加原始用户问题到会话
        self.add_message("user", question)

        logger.info(f"Processing question with streaming: {question}")
        try:
            # 获取检索到的文档，增加检索数量以提高相关性
            retriever = self.db.as_retriever(search_kwargs={"k": 6})
            docs = retriever.invoke(question)
            
            # 按相关性分数排序文档（如果有分数）
            if hasattr(docs[0], 'metadata') and 'score' in docs[0].metadata:
                docs = sorted(docs, key=lambda x: x.metadata.get('score', 0), reverse=True)
            
            # 准备检索到的文档内容作为参考，添加文档来源信息
            context_parts = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', '未知来源') if hasattr(doc, 'metadata') else '未知来源'
                context_parts.append(f"文档{i+1}（来源：{source}）:\n{doc.page_content}")
            context = "\n\n".join(context_parts)
            
            # 获取会话历史消息作为上下文
            messages = self.get_session_messages()
            
            # 移除最后一条用户消息，因为我们将添加带有参考资料的增强版本
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            
            # 添加当前问题和检索到的文档内容作为单独的用户消息
            messages.append({"role": "user", "content": f"问题：{question}\n\n参考资料（按相关性排序）：\n{context}"})
            
            # 添加提示，引导模型关注相关内容
            if len(docs) > 0:
                logger.info(f"Retrieved {len(docs)} documents for question: {question}")
            else:
                logger.warning(f"No documents retrieved for question: {question}")            
            
            # 使用流式输出
            response = ""
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    response += chunk.content
                    yield response, docs
            
            # 添加助手回答到会话
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

    # 设置日志级别
    if args.debug:
        # 使用loguru的方式设置日志级别
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

    # 加载或创建数据库
    if args.create_db or not rag.load_database(force_recreate=args.force_recreate):
        if not args.force_recreate:  # 避免重复创建
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
