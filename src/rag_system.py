import os
import logging
import time
from typing import List, Dict, Any, Optional
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.zhipu_client import ZhipuAIClient
from src.config import Config

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG系统协调器：整合所有模块提供完整的问答服务"""

    def __init__(self, docs_dir: str = "documents"):
        self.docs_dir = docs_dir
        self.document_processor = None
        self.vector_store = None
        self.llm_client = None
        self.is_initialized = False
        self.initialization_time = None

    def initialize(self, rebuild_index: bool = False) -> bool:
        """初始化RAG系统"""
        start_time = time.time()

        try:
            logger.info("开始初始化RAG系统...")

            # 1. 初始化配置
            Config.validate()

            # 2. 初始化各组件
            self.document_processor = DocumentProcessor(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )

            self.vector_store = VectorStore(
                model_path=Config.EMBEDDING_MODEL_PATH,
                model_name=Config.EMBEDDING_MODEL_NAME
            )

            self.llm_client = ZhipuAIClient()

            # 3. 检查是否需要重建索引
            if rebuild_index or not self._index_exists():
                logger.info("构建新的向量索引...")
                self._build_vector_index()
            else:
                logger.info("加载现有向量索引...")
                self.vector_store.load_index(Config.VECTOR_STORE_PATH)

            self.is_initialized = True
            self.initialization_time = time.time() - start_time

            logger.info(f"RAG系统初始化完成，耗时: {self.initialization_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"RAG系统初始化失败: {e}")
            self.is_initialized = False
            return False

    def _index_exists(self) -> bool:
        """检查向量索引是否存在"""
        index_file = os.path.join(Config.VECTOR_STORE_PATH, "index.faiss")
        chunks_file = os.path.join(Config.VECTOR_STORE_PATH, "chunks.pkl")
        return os.path.exists(index_file) and os.path.exists(chunks_file)

    def _build_vector_index(self):
        """构建向量索引"""
        # 加载和分割文档
        documents = self.document_processor.load_documents(self.docs_dir)
        if not documents:
            raise ValueError(f"在目录 {self.docs_dir} 中未找到文档")

        chunks = self.document_processor.split_documents(documents)

        # 构建向量索引
        self.vector_store.build_index(chunks)

        # 保存索引
        self.vector_store.save_index(Config.VECTOR_STORE_PATH)

    def ask_question(self,
                     question: str,
                     top_k: int = None,
                     include_references: bool = True) -> Dict[str, Any]:
        """
        回答问题

        Args:
            question: 用户问题
            top_k: 检索的文档数量
            include_references: 是否包含参考文档

        Returns:
            包含回答和元数据的字典
        """
        if not self.is_initialized:
            return {
                "success": False,
                "answer": "系统未初始化，请先调用initialize()方法",
                "error": "System not initialized"
            }

        start_time = time.time()

        try:
            top_k = top_k or Config.MAX_RETRIEVAL_DOCS

            # 1. 检索相关文档
            retrieval_start = time.time()
            references = self.vector_store.search(question, top_k=top_k)
            retrieval_time = time.time() - retrieval_start

            # 2. 构建上下文
            context = ""
            if references and include_references:
                context_parts = []
                for i, ref in enumerate(references):
                    context_parts.append(
                        f"[文档 {i + 1}] 来源: {ref['file_name']}\n"
                        f"内容: {ref['content']}\n"
                        f"相似度: {ref['similarity_score']:.4f}"
                    )
                context = "\n\n".join(context_parts)

            # 3. 生成回答
            generation_start = time.time()
            if context:
                answer = self.llm_client.rag_chat(question, context)
            else:
                answer = self.llm_client.chat(question)
            generation_time = time.time() - generation_start

            total_time = time.time() - start_time

            # 构建响应
            response = {
                "success": True,
                "question": question,
                "answer": answer,
                "references": references if include_references else [],
                "context_used": context if include_references else "",
                "performance": {
                    "total_time": total_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "references_count": len(references)
                }
            }

            logger.info(f"问题回答完成: '{question}' (总耗时: {total_time:.2f}s)")
            return response

        except Exception as e:
            logger.error(f"回答问题失败: {e}")
            return {
                "success": False,
                "question": question,
                "answer": f"抱歉，回答问题时报错: {str(e)}",
                "error": str(e),
                "performance": {
                    "total_time": time.time() - start_time
                }
            }

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        if not self.is_initialized:
            return {"status": "未初始化"}

        # 获取文档信息
        documents = self.document_processor.load_documents(self.docs_dir)
        chunks = self.document_processor.split_documents(documents) if documents else []

        return {
            "status": "运行中",
            "initialization_time": f"{self.initialization_time:.2f}秒",
            "documents_count": len(documents) if documents else 0,
            "chunks_count": len(chunks),
            "vector_index_size": len(self.vector_store.chunks) if self.vector_store.chunks else 0,
            "config": {
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "max_retrieval_docs": Config.MAX_RETRIEVAL_DOCS,
                "embedding_model": Config.EMBEDDING_MODEL_NAME
            }
        }