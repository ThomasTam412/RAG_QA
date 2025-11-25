import streamlit as st
import os
import sys
import time
import logging
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_system import RAGSystem
from src.config import Config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 初始化会话状态
def initialize_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False


def initialize_rag_system(rebuild_index: bool = False):
    """初始化RAG系统"""
    try:
        with st.spinner("正在初始化RAG系统，这可能需要一些时间..."):
            rag = RAGSystem()
            success = rag.initialize(rebuild_index=rebuild_index)

            if success:
                st.session_state.rag_system = rag
                st.session_state.system_initialized = True
                return True
            else:
                st.error("RAG系统初始化失败")
                return False
    except Exception as e:
        st.error(f"初始化过程中出错: {str(e)}")
        return False


def display_conversation():
    """显示对话历史"""
    for i, exchange in enumerate(st.session_state.conversation_history):
        with st.container():
            # 用户问题
            with st.chat_message("user"):
                st.write(f"**Q:** {exchange['question']}")

            # 系统回答
            with st.chat_message("assistant"):
                st.write(f"**A:** {exchange['answer']}")

                # 显示参考文档（可折叠）
                if exchange.get('references'):
                    with st.expander(f"参考文档 ({len(exchange['references'])} 个)"):
                        for j, ref in enumerate(exchange['references']):
                            st.markdown(
                                f"**文档 {j + 1}** - `{ref['file_name']}` (相似度: `{ref['similarity_score']:.4f}`)")
                            st.text(f"{ref['content'][:200]}...")

                # 显示性能信息
                if exchange.get('performance'):
                    perf = exchange['performance']
                    st.caption(f"检索: {perf['retrieval_time']:.2f}s | "
                               f"生成: {perf['generation_time']:.2f}s | "
                               f"总计: {perf['total_time']:.2f}s")

            st.markdown("---")


def main():
    # 初始化会话状态
    initialize_session_state()

    # 页面标题
    st.title("🤖 RAG智能问答系统")
    st.markdown("基于本地文档的智能问答系统，使用检索增强生成技术")

    # 侧边栏
    with st.sidebar:
        st.header("系统配置")

        # 初始化选项
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 初始化系统", use_container_width=True):
                if initialize_rag_system(rebuild_index=False):
                    st.rerun()

        with col2:
            if st.button("🔄 重建索引", use_container_width=True):
                if initialize_rag_system(rebuild_index=True):
                    st.rerun()

        st.markdown("---")

        # 系统信息
        if st.session_state.system_initialized:
            st.header("系统状态")
            info = st.session_state.rag_system.get_system_info()

            st.metric("文档数量", info["documents_count"])
            st.metric("文本块数量", info["chunks_count"])
            st.metric("向量索引大小", info["vector_index_size"])

            with st.expander("详细配置"):
                st.json(info["config"])

        st.markdown("---")
        st.header("关于")
        st.markdown("""
        - **技术栈**: Streamlit + FAISS + Sentence-BERT + 智谱GLM
        - **文档格式**: 支持 TXT 和 PDF
        - **嵌入模型**: all-MiniLM-L6-v2
        - **LLM**: 智谱AI GLM-4
        """)

    # 主界面
    if not st.session_state.system_initialized:
        st.info("👈 请先点击侧边栏的「初始化系统」按钮来启动RAG系统")

        # 显示文档预览
        if os.path.exists("documents"):
            documents = [f for f in os.listdir("documents") if f.endswith(('.txt', '.pdf'))]
            if documents:
                st.subheader("文档库预览")
                st.write(f"检测到 {len(documents)} 个文档文件:")
                for doc in documents:
                    st.write(f"- {doc}")

        return

    # 问答界面
    st.header("💬 智能问答")

    # 问题输入
    question = st.chat_input("请输入您的问题...")

    # 高级设置
    with st.expander("高级设置"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("检索文档数量", 1, 10, Config.MAX_RETRIEVAL_DOCS)
        with col2:
            include_refs = st.checkbox("显示参考文档", value=True)

    # 处理用户问题
    if question:
        # 添加用户消息到对话历史
        st.session_state.conversation_history.append({
            "question": question,
            "answer": "思考中...",
            "timestamp": datetime.now()
        })

        # 显示用户消息
        with st.chat_message("user"):
            st.write(question)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索文档和生成回答..."):
                response = st.session_state.rag_system.ask_question(
                    question,
                    top_k=top_k,
                    include_references=include_refs
                )

            # 更新对话历史
            st.session_state.conversation_history[-1].update({
                "answer": response["answer"],
                "references": response.get("references", []),
                "performance": response.get("performance", {}),
                "success": response.get("success", False)
            })

            # 显示回答
            st.write(response["answer"])

            # 显示参考文档
            if include_refs and response.get("references"):
                with st.expander(f"参考文档 ({len(response['references'])} 个)"):
                    for i, ref in enumerate(response['references']):
                        st.markdown(f"**文档 {i + 1}** - `{ref['file_name']}` "
                                    f"(相似度: `{ref['similarity_score']:.4f}`)")
                        st.text(f"{ref['content']}")

            # 显示性能信息
            if response.get("performance"):
                perf = response["performance"]
                st.caption(f"⏱️ 检索: {perf['retrieval_time']:.2f}s | "
                           f"生成: {perf['generation_time']:.2f}s | "
                           f"总计: {perf['total_time']:.2f}s")

    # 显示对话历史（排除当前正在处理的问题）
    if len(st.session_state.conversation_history) > 0:
        st.header("📜 对话历史")

        # 只显示已完成的对话
        completed_conversations = [
            conv for conv in st.session_state.conversation_history
            if conv.get("answer") != "思考中..."
        ]

        if completed_conversations:
            for i, exchange in enumerate(completed_conversations):
                with st.container():
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.write(f"**{i + 1}.**")
                    with col2:
                        st.write(f"**Q:** {exchange['question']}")
                        st.write(f"**A:** {exchange['answer']}")

                        if exchange.get('references'):
                            with st.expander(f"参考文档 ({len(exchange['references'])} 个)"):
                                for j, ref in enumerate(exchange['references']):
                                    st.markdown(f"**文档 {j + 1}** - `{ref['file_name']}` "
                                                f"(相似度: `{ref['similarity_score']:.4f}`)")
                                    st.text(
                                        f"{ref['content'][:200]}..." if len(ref['content']) > 200 else ref['content'])

                    st.markdown("---")
        else:
            st.info("暂无对话历史")


if __name__ == "__main__":
    main()