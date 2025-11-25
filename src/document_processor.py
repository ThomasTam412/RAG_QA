import os
import re
import logging
from typing import List, Dict, Any

# 尝试多种PDF库导入
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("需要安装 pypdf 或 PyPDF2 库来处理PDF文件")
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器：负责加载和分割文档"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, docs_dir: str) -> List[Dict[str, Any]]:
        """加载目录中的所有文档"""
        documents = []

        if not os.path.exists(docs_dir):
            logger.error(f"文档目录不存在: {docs_dir}")
            return documents

        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)
            if filename.endswith('.txt'):
                content = self._load_txt(file_path)
                if content:
                    documents.append({
                        'file_name': filename,
                        'content': content,
                        'type': 'txt'
                    })
            elif filename.endswith('.pdf'):
                content = self._load_pdf(file_path)
                if content:
                    documents.append({
                        'file_name': filename,
                        'content': content,
                        'type': 'pdf'
                    })

        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _load_txt(self, file_path: str) -> str:
        """加载txt文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            logger.debug(f"加载TXT文件: {os.path.basename(file_path)}")
            return content
        except Exception as e:
            logger.error(f"加载TXT文件失败 {file_path}: {e}")
            return ""

    def _load_pdf(self, file_path: str) -> str:
        """加载PDF文件"""
        try:
            text = ""
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            logger.debug(f"加载PDF文件: {os.path.basename(file_path)}")
            return text.strip()
        except Exception as e:
            logger.error(f"加载PDF文件失败 {file_path}: {e}")
            return ""

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分割文档为文本块"""
        all_chunks = []

        for doc in documents:
            chunks = self._split_text(doc['content'])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'file_name': doc['file_name'],
                    'content': chunk,
                    'chunk_id': i,
                    'type': doc['type']
                })

        logger.info(f"将文档分割为 {len(all_chunks)} 个文本块")
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """分割单个文本为块"""
        # 按句子分割（简单实现）
        sentences = re.split(r'(?<=[。！？\.!?])\s*', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果当前块加上新句子不会超过限制
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 开始新块（包含重叠）
                if self.chunk_overlap > 0 and chunks:
                    last_chunk = chunks[-1]
                    overlap_start = max(0, len(last_chunk) - self.chunk_overlap)
                    current_chunk = last_chunk[overlap_start:] + " " + sentence + " "
                else:
                    current_chunk = sentence + " "

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks