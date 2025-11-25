import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """配置管理类"""

    # API配置
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

    # 模型配置
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models")

    # 向量存储配置
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # 应用配置
    MAX_RETRIEVAL_DOCS = int(os.getenv("MAX_RETRIEVAL_DOCS", "3"))

    @classmethod
    def validate(cls):
        """验证必要配置"""
        if not cls.ZHIPU_API_KEY:
            raise ValueError("请设置 ZHIPU_API_KEY 环境变量")

        # 确保目录存在（在Streamlit Cloud上这些目录应该已经存在）
        os.makedirs(cls.EMBEDDING_MODEL_PATH, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_PATH, exist_ok=True)

        print("✓ 配置验证通过")