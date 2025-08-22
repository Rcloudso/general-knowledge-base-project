import os
import time
from db_manager import DatabaseManager

def test_database_manager():
    # 使用测试数据库文件
    test_db_path = "./test_sessions.db"
    
    # 如果测试数据库已存在，先删除它
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # 创建数据库管理器
    db_manager = DatabaseManager(test_db_path)
    
    # 测试创建会话
    session_id = f"test_session_{int(time.time())}"
    system_message = "你是一个有用的AI助手，根据提供的上下文回答问题。"
    assert db_manager.create_session(session_id, system_message)
    
    # 测试会话存在性检查
    assert db_manager.session_exists(session_id)
    assert not db_manager.session_exists("non_existent_session")
    
    # 测试添加消息
    assert db_manager.add_message(session_id, "user", "你好，这是一个测试问题。")
    assert db_manager.add_message(session_id, "assistant", "你好！我是AI助手，很高兴为你提供帮助。")
    
    # 测试获取会话消息
    messages = db_manager.get_session_messages(session_id)
    assert len(messages) == 3  # 系统消息 + 用户消息 + 助手消息
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    
    # 测试列出会话
    sessions = db_manager.list_sessions()
    assert session_id in sessions
    
    # 测试创建另一个会话
    another_session_id = f"another_session_{int(time.time())}"
    assert db_manager.create_session(another_session_id, system_message)
    
    # 再次测试列出会话
    sessions = db_manager.list_sessions()
    assert session_id in sessions
    assert another_session_id in sessions
    
    # 测试删除会话
    assert db_manager.delete_session(session_id)
    assert not db_manager.session_exists(session_id)
    
    # 清理
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    print("所有测试通过！")

if __name__ == "__main__":
    test_database_manager()