import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# 尝试导入FAISS，如果失败则提供明确错误信息
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError as e:
    logging.error(f"FAISS导入失败: {e}")
    logging.error("请确保已安装faiss-cpu: pip install faiss-cpu")
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储：负责文本嵌入、向量索引构建和相似性搜索"""

    def __init__(self, model_path: str = "./models", model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.index = None
        self.chunks = []
        self.model_path = model_path
        self.model_name = model_name
        self._initialize_model()

    def _initialize_model(self):
        """初始化嵌入模型"""
        try:
            # 尝试从本地路径加载模型
            local_model_path = os.path.join(self.model_path, self.model_name)
            if os.path.exists(local_model_path):
                logger.info(f"从本地加载模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path)
            else:
                logger.info(f"本地模型未找到，从网络下载: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                # 保存到本地以备后续使用
                os.makedirs(self.model_path, exist_ok=True)
                self.model.save(local_model_path)
                logger.info(f"模型已保存到本地: {local_model_path}")

            logger.info(f"嵌入模型初始化完成: {self.model_name}")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def build_index(self, chunks: List[Dict[str, Any]]):
        """构建向量索引"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS不可用，无法构建向量索引")

        if not self.model:
            raise RuntimeError("嵌入模型未初始化")

        logger.info("开始构建向量索引...")

        # 提取文本内容
        texts = [chunk['content'] for chunk in chunks]
        self.chunks = chunks

        # 生成嵌入向量
        logger.info("生成文本嵌入...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # 创建FAISS索引（使用内积相似度）
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度

        # 归一化向量以便使用内积相似度（等同于余弦相似度）
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))

        logger.info(f"向量索引构建完成: {len(chunks)} 个文档块，维度 {dimension}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        if not self.index:
            raise RuntimeError("向量索引未构建，请先调用 build_index()")

        # 生成查询嵌入
        query_embedding = self.model.encode([query])

        # 归一化查询向量
        faiss.normalize_L2(query_embedding)

        # 执行搜索
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk_info = self.chunks[idx].copy()
                chunk_info['similarity_score'] = float(score)
                results.append(chunk_info)

        logger.debug(f"搜索查询: '{query}'，返回 {len(results)} 个结果")
        return results

    def save_index(self, save_path: str):
        """保存向量索引和文档块"""
        if not self.index:
            raise RuntimeError("没有可保存的索引")

        os.makedirs(save_path, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))

        # 保存文档块信息
        with open(os.path.join(save_path, "chunks.pkl"), 'wb') as f:
            pickle.dump(self.chunks, f)

        logger.info(f"向量索引已保存到: {save_path}")

    def load_index(self, load_path: str):
        """加载向量索引和文档块"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS不可用，无法加载向量索引")

        # 加载FAISS索引
        index_path = os.path.join(load_path, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        self.index = faiss.read_index(index_path)

        # 加载文档块信息
        chunks_path = os.path.join(load_path, "chunks.pkl")
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        logger.info(f"向量索引已从 {load_path} 加载: {len(self.chunks)} 个文档块")