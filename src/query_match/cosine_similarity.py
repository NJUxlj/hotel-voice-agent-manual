import numpy as np
from src.models.model import ZhipuEmbedding

class CosineSimilarity:
    def __init__(self):
        self.embedding_model = ZhipuEmbedding()
    def calculate_similarity(self, text1, text2):
        # 获取文本的嵌入向量
        embedding1 = self.embedding_model.get_single_embedding(text1)
        embedding2 = self.embedding_model.get_single_embedding(text2)
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity[0][0]