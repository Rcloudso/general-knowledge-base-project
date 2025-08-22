import streamlit as st
import os
import time
from rag_core import RAGCore

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="RAG知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
def initialize_rag_core():
    """Initializes or retrieves the RAGCore instance from session state."""
    if "rag_core" not in st.session_state:
        st.session_state.rag_core = RAGCore()
    return st.session_state.rag_core

def load_existing_db(rag_core_instance):
    """尝试加载现有数据库并更新状态。"""
    if rag_core_instance.load_database():
        st.session_state.db_created = True
        st.sidebar.success("已成功加载现有知识库！")
    else:
        st.sidebar.info("未找到现有知识库，请创建一个新的知识库。")
        
def create_new_session():
    """创建新的会话"""
    rag_core = initialize_rag_core()
    session_id = rag_core.create_session()
    st.session_state.messages = []
    st.sidebar.success(f"已创建新会话: {session_id}")
    return session_id

def switch_session(session_id):
    """切换到指定会话"""
    rag_core = initialize_rag_core()
    if rag_core.switch_session(session_id):
        # 加载会话消息到Streamlit界面
        messages = rag_core.get_session_messages()
        # 过滤掉系统消息，只显示用户和助手的消息
        st.session_state.messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] != "system"
        ]
        st.sidebar.success(f"已切换到会话: {session_id}")
        return True
    return False

# --- Sidebar UI ---
st.sidebar.title("📚 知识库设置")

# Input for the documents folder path
if "folder_path" not in st.session_state:
    st.session_state.folder_path = ""

folder_path = st.sidebar.text_input(
    "文档文件夹路径:", 
    value=st.session_state.folder_path,
    help="包含文档文件的文件夹路径 (.pdf, .txt, .doc, .docx)。"
)

# 添加选择文件夹按钮
if st.sidebar.button("📂 选择文件夹"):
    # 使用PowerShell脚本打开文件夹选择对话框
    import subprocess
    try:
        # PowerShell脚本，用于打开文件夹选择对话框并返回选择的路径
        ps_script = '''
        Add-Type -AssemblyName System.Windows.Forms
        $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
        $folderBrowser.Description = "选择包含文档的文件夹"
        $folderBrowser.RootFolder = "MyComputer"
        if ($folderBrowser.ShowDialog() -eq "OK") {
            $folderBrowser.SelectedPath
        }
        '''
        
        # 执行PowerShell脚本
        result = subprocess.run(["powershell", "-Command", ps_script], 
                               capture_output=True, text=True, encoding='utf-8')
        
        # 获取选择的文件夹路径
        selected_path = result.stdout.strip()
        
        if selected_path:
            st.session_state.folder_path = selected_path
            st.rerun()  # 重新运行应用以更新UI
    except Exception as e:
        st.sidebar.error(f"选择文件夹时出错: {e}")

# 使用会话状态中的文件夹路径
folder_path = st.session_state.folder_path

# Button to create the knowledge base
if st.sidebar.button("📚 创建知识库"):
    if folder_path and os.path.isdir(folder_path):
        with st.spinner(f"正在处理'{folder_path}'中的文件... 这可能需要一些时间。"):
            try:
                rag_core = initialize_rag_core()
                rag_core.directory_path = folder_path
                rag_core.create_database()
                st.session_state.db_created = True
                st.sidebar.success("知识库创建成功！")
            except Exception as e:
                st.sidebar.error(f"发生错误: {e}")
    elif not folder_path:
        st.sidebar.warning("请输入文件夹路径。")
    else:
        st.sidebar.error("提供的路径不是有效的目录。")

st.sidebar.markdown("---")

# --- 会话管理 UI ---
st.sidebar.title("💬 会话管理")

# 初始化RAG Core
rag_core = initialize_rag_core()

# 获取当前会话ID
current_session_id = rag_core.current_session_id

# 新建会话按钮
if st.sidebar.button("➕ 新建会话"):
    create_new_session()
    st.rerun()

# 显示会话列表
st.sidebar.subheader("会话列表")
sessions = rag_core.list_sessions()

if sessions:
    # 格式化会话列表，显示创建时间
    for session_id, created_at in sessions.items():
        created_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(created_at))
        session_label = f"{created_time} - {session_id}"
        
        # 为当前会话添加标记
        if session_id == current_session_id:
            session_label = f"✓ {session_label}"
        
        # 会话选择和删除按钮
        col1, col2 = st.sidebar.columns([4, 1])
        
        # 切换会话按钮
        if col1.button(session_label, key=f"session_{session_id}"):
            if switch_session(session_id):
                st.rerun()
        
        # 删除会话按钮
        if col2.button("🗑️", key=f"delete_{session_id}"):
            if rag_core.delete_session(session_id):
                st.sidebar.success(f"已删除会话: {session_id}")
                # 如果删除的是当前会话，创建一个新会话
                if session_id == current_session_id:
                    create_new_session()
                st.rerun()
else:
    st.sidebar.info("没有会话记录，开始提问将自动创建新会话。")

st.sidebar.markdown("---")

# --- Main Page UI ---
st.title("🤖 知识库问答系统")
st.markdown("创建知识库后，您可以在此处提问关于您文档的问题。")

# Initialize RAG Core and attempt to load DB on first run
rag_core = initialize_rag_core()
if "db_created" not in st.session_state:
    load_existing_db(rag_core)

# 确保有活跃会话
if not rag_core.current_session_id and rag_core.list_sessions():
    # 如果有会话但没有活跃会话，选择最新的一个
    sessions = rag_core.list_sessions()
    latest_session = max(sessions.items(), key=lambda x: x[1])[0]
    switch_session(latest_session)
elif not rag_core.current_session_id:
    # 如果没有任何会话，创建新会话
    create_new_session()

# Main chat interface
if st.session_state.get("db_created", False):
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        # 从当前会话加载消息
        messages = rag_core.get_session_messages()
        # 过滤掉系统消息
        st.session_state.messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] != "system"
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("请输入您的问题"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("思考中..."):
                try:
                    # 使用流式输出，传递当前会话ID
                    full_response = ""
                    sources = []
                    
                    # 显示流式回答
                    for partial_answer, docs in rag_core.query_stream(prompt, rag_core.current_session_id):
                        if docs and not sources:  # 只在第一次获取源文档
                            sources = docs
                        
                        # 更新显示
                        full_response = partial_answer
                        message_placeholder.markdown(full_response)
                    
                    # 添加源文件信息
                    if sources:
                        source_files = list(set([os.path.basename(s.metadata['source']) for s in sources]))
                        response_with_sources = f"{full_response}\n\n**Sources:**\n"
                        for src in source_files:
                            response_with_sources += f"- `{src}`\n"
                        
                        # 更新最终显示
                        message_placeholder.markdown(response_with_sources)
                        st.session_state.messages.append({"role": "assistant", "content": response_with_sources})
                    else:
                        # 如果没有源文件，直接保存回答
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_message = f"抱歉，发生了错误: {e}"
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("请使用侧边栏创建或加载知识库以开始使用。")
