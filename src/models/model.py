import os  
import json  
import numpy as np  
import requests  
from typing import List, Dict, Any, Optional  
import torch  
import faiss  
from tqdm import tqdm  
import time  
from zhipuai import ZhipuAI  # 智谱AI的官方Python客户端  


ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")  # 从环境变量获取智谱API密钥

class ZhipuEmbedding:  
    """  
    使用智谱AI API的文本嵌入生成器  
    """  
    
    def __init__(
        self, 
        api_key: str = ZHIPU_API_KEY, 
        model_name: str = "embedding-2"):  
        """  
        初始化智谱嵌入生成器  
        
        参数:  
            api_key: 智谱API密钥  
            model_name: 嵌入模型名称，默认为'embedding-2'  
        """  
        self.api_key = api_key  
        self.model_name = model_name  
        self.client = ZhipuAI(api_key=api_key)  
        self.embedding_dimension = 1024  # 智谱embedding-2模型的维度  
        
        # 请求限速设置  
        self.request_interval = 0.5  # 请求间隔(秒)  
        self.last_request_time = 0  
        
        print(f"初始化智谱API嵌入生成器 (模型: {model_name})")  
    
    def _rate_limit(self):  
        """简单的请求限速实现"""  
        current_time = time.time()  
        time_since_last = current_time - self.last_request_time  
        
        if time_since_last < self.request_interval:  
            sleep_time = self.request_interval - time_since_last  
            time.sleep(sleep_time)  
            
        self.last_request_time = time.time()  
    
    def get_embeddings(self, texts: List[str], batch_size: int = 10) -> np.ndarray:  
        """  
        获取文本列表的嵌入向量  
        
        参数:  
            texts: 文本列表  
            batch_size: 批处理大小  
            
        返回:  
            嵌入向量数组  
        """  
        all_embeddings = []  
        
        # 分批处理  
        for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量"):  
            batch = texts[i:i+batch_size]  
            batch_embeddings = []  
            
            for text in batch:  
                self._rate_limit()  # 限速  
                try:  
                    # 调用智谱API  
                    response = self.client.embeddings.create(  
                        model=self.model_name,  
                        input=text  
                    )  
                    
                    # 提取嵌入向量  
                    embedding = response.data[0].embedding  
                    batch_embeddings.append(embedding)  
                    
                except Exception as e:  
                    print(f"获取嵌入向量失败: {str(e)}")  
                    # 返回零向量作为后备  
                    batch_embeddings.append([0.0] * self.embedding_dimension)  
            
            all_embeddings.extend(batch_embeddings)  
        
        # 转换为numpy数组  
        return np.array(all_embeddings)  
    
    def get_single_embedding(self, text: str) -> np.ndarray:  
        """  
        获取单个文本的嵌入向量  
        
        参数:  
            text: 输入文本  
            
        返回:  
            嵌入向量  
        """  
        self._rate_limit()  # 限速  
        try:  
            # 调用智谱API  
            response = self.client.embeddings.create(  
                model=self.model_name,  
                input=text  
            )  
            
            # 提取嵌入向量  
            embedding = response.data[0].embedding  # List[float]
            return np.array([embedding])  
            
        except Exception as e:  
            print(f"获取嵌入向量失败: {str(e)}")  
            # 返回零向量作为后备  
            return np.array([[0.0] * self.embedding_dimension])  
    
    def get_dimension(self) -> int:  
        """  
        返回嵌入向量的维度  
        
        返回:  
            嵌入向量维度  
        """  
        return self.embedding_dimension  
    
    
    




if __name__ == "__main__":
    # 测试代码
    embedding_model = ZhipuEmbedding()
    texts = ["这是一个测试文本", "这是另一个测试文本"]
    
    # 获取嵌入向量
    embeddings = embedding_model.get_embeddings(texts)
    print("嵌入向量形状:", embeddings.shape)
    print("嵌入向量:", embeddings)
    
    
    
# 运行： python -m src.models.model