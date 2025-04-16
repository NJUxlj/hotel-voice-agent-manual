import numpy as np
from src.models.model import ZhipuEmbedding


class L2Distance:
    def __init__(self):
        self.embedding_model = ZhipuEmbedding()
    
    def calculate_similarity(self, text1, text2):
        # 获取文本的嵌入向量
        embedding1 = self.embedding_model.get_single_embedding(text1)
        embedding2 = self.embedding_model.get_single_embedding(text2)
        # 计算L2距离
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance