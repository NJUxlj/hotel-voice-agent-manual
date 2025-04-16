# 上海旅游多智能体系统  
import os  
import json  
import numpy as np  
from typing import Dict, List, Any, Optional, Tuple  
import zhipuai  
from zhipuai import ZhipuAI  
from serpapi import GoogleSearch  
from datetime import datetime  


# LangChain imports  
from langchain_community.vectorstores.chroma import Chroma  
from langchain_core.documents import Document  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.runnables import RunnablePassthrough  
from langchain_core.prompts import ChatPromptTemplate  
# from langchain_community.llms import ChatZhipuAI, ZhipuAIEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  



from src.models.model import ZhipuEmbedding



# 设置API密钥（请替换为您的实际密钥）  
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "your_key")  # 替换为您的智谱API密钥  
SERPAPI_API_KEY = "YOUR_SERPAPI_API_KEY"  # 替换为您的SerpAPI密钥  

# 初始化智谱AI客户端  
client = ZhipuAI(api_key=ZHIPU_API_KEY)  

# 文件路径定义  
DATA_DIR = "data"  
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db.json")  
SHANGHAI_TOURISM_DOCS = os.path.join(DATA_DIR, "shanghai_tourism_docs.txt")  

# 确保数据目录存在  
os.makedirs(DATA_DIR, exist_ok=True)  

# 定义简单的向量数据库类  
class SimpleVectorDB:  
    """简单的向量数据库实现，用于存储文档嵌入和检索"""  
    
    def __init__(self, file_path=VECTOR_DB_PATH):  
        self.file_path = file_path  
        self.db = self._load_db()  
    
    def _load_db(self):  
        """从文件加载向量数据库"""  
        if os.path.exists(self.file_path):  
            try:  
                with open(self.file_path, 'r', encoding='utf-8') as f:  
                    return json.load(f)  
            except Exception as e:  
                print(f"加载向量数据库失败: {e}")  
                return {"documents": [], "embeddings": []}  
        else:  
            return {"documents": [], "embeddings": []}  
    
    def save_db(self):  
        """将向量数据库保存到文件"""  
        with open(self.file_path, 'w', encoding='utf-8') as f:  
            json.dump(self.db, f, ensure_ascii=False, indent=2)  
    
    def add_document(self, document: str, embedding: List[float]):  
        """添加文档及其嵌入向量到数据库"""  
        self.db["documents"].append(document)  
        self.db["embeddings"].append(embedding)  
        self.save_db()  
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:  
        """根据查询嵌入搜索最相似的文档"""  
        if not self.db["embeddings"]:  
            return []  
        
        # 计算余弦相似度  
        similarities = []  
        for doc_embedding in self.db["embeddings"]:  
            similarity = self._cosine_similarity(query_embedding, doc_embedding)  
            similarities.append(similarity)  
        
        # 获取top-k最相似的文档  
        top_indices = np.argsort(similarities)[-top_k:][::-1]  
        results = [(self.db["documents"][i], similarities[i]) for i in top_indices]  
        return results  
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:  
        """计算两个向量之间的余弦相似度"""  
        vec1 = np.array(vec1)  
        vec2 = np.array(vec2)  
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))  


# 执行Agent基类  
class BaseAgent:  
    """智能体基类，提供基本方法"""  
    
    def __init__(self, name: str):  
        self.name = name  
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:  
        """运行智能体"""  
        raise NotImplementedError("Subclasses must implement run method")  


# RAG执行Agent  
class RAGAgent(BaseAgent):  
    """负责检索增强生成的执行智能体"""  
    
    def __init__(self, name: str = "RAG智能体"):  
        super().__init__(name)  
        self.vector_db = SimpleVectorDB()  
        self.embedding_model = ZhipuEmbedding(api_key=ZHIPU_API_KEY)
        self._init_data()  
        
        # 用于LangChain的API密钥设置  
        self.api_key = os.environ.get("ZHIPU_API_KEY", "YOUR_ZHIPU_API_KEY")  
        
        # 存储LangChain临时数据的目录  
        self.persist_directory = os.path.join(os.getcwd(), "chroma_db")  
        
        # LangChain组件  
        self.langchain_initialized = False  
        self.chroma_db = None  
        self.zhipu_llm = None  
        self.retriever = None  
        self.rag_chain = None  
    
    def _init_data(self):  
        """初始化数据，如果向量数据库为空则导入上海旅游知识"""  
        if not self.vector_db.db["documents"]:  
            self._populate_knowledge_base()  
    
    def _populate_knowledge_base(self):  
        """向知识库中添加上海旅游知识"""  
        # 检查是否存在上海旅游文档文件  
        if not os.path.exists(SHANGHAI_TOURISM_DOCS):  
            # 如果不存在,创建一些示例数据  
            sample_data = [
                "上海，简称\"沪\"，是中国的一个直辖市，国家中心城市，超大城市，也是国际经济、金融、贸易、航运、科技创新中心。",  
                "上海有许多著名的旅游景点，包括外滩、东方明珠、上海迪士尼、豫园、南京路、城隍庙等。",  
                "外滩是上海最著名的景点之一，沿黄浦江而建，有52幢风格各异的历史建筑，被称为\"万国建筑博览群\"。",  
                "东方明珠电视塔是上海的标志性建筑，高468米，是亚洲第一、世界第三高塔。",  
                "上海迪士尼度假区于2016年6月16日正式开园，是中国内地首个迪士尼主题乐园。",  
                "豫园建于明代嘉靖年间，是江南古典园林的代表作品之一，有\"城市山林\"之称。",  
                "南京路是中国第一条商业街，也是上海开埠后最早建立的一条商业街，被誉为\"中华商业第一街\"。",  
                "上海的四季分明，春秋两季较短，冬夏较长。最佳旅游时间为3-5月和9-11月。",  
                "上海菜，也称沪菜，是中国八大菜系之一，以本帮菜为代表，特点是口味偏甜。",  
                "上海交通发达，有地铁、公交、出租车等多种交通方式，地铁是游览上海的最佳选择之一。",  
                "上海的著名美食包括小笼包、生煎包、蟹壳黄、红烧肉、八宝饭等。",  
                "上海话属于吴语的一种，是上海地区的主要语言，但普通话在上海也很普及。",  
                "上海博物馆是中国首批国家一级博物馆，收藏了大量珍贵文物。",  
                "上海科技馆是中国规模最大的科技馆之一，适合亲子游览。",  
                "上海野生动物园位于浦东新区，是中国规模最大的野生动物园之一。",  
                "上海环球金融中心是上海的地标性建筑之一，观光厅可俯瞰整个上海。",  
                "七宝古镇是上海著名的历史文化古镇，有\"沪上方浜\"之称。",  
                "朱家角古镇是上海的四大历史文化名镇之一，以水乡风貌著称。",  
                "田子坊是上海的创意产业集聚区，由一组石库门里弄改造而成。",  
                "新天地是上海的时尚休闲街区，由旧时上海的石库门建筑改造而成。"  
            ]  
            
            with open(SHANGHAI_TOURISM_DOCS, 'w', encoding='utf-8') as f:  
                f.write('\n'.join(sample_data))  
            
            # 将样本数据加入向量数据库  
            for doc in sample_data:  
                embedding = self._get_embedding(doc)  
                self.vector_db.add_document(doc, embedding)  
        else:  
            # 从文件加载数据  
            with open(SHANGHAI_TOURISM_DOCS, 'r', encoding='utf-8') as f:  
                lines = f.readlines()  
                
            # 检查向量数据库是否为空  
            if not self.vector_db.db["documents"]:  
                for line in lines:  
                    line = line.strip()  
                    if line:  
                        embedding = self._get_embedding(line)  
                        self.vector_db.add_document(line, embedding)  
    
    def _get_embedding(self, text: str) -> List[float]:  
        """使用智谱AI获取文本的嵌入向量"""  
        # try:  
        #     # 调用智谱AI的嵌入接口  
        #     response = client.embeddings.create(  
        #         model="embedding-2",  
        #         input=text  
        #     )  
        #     return response.data[0].embedding  
        # except Exception as e:  
        #     print(f"获取 embedding-2 嵌入向量失败: {e}")  
        #     # 返回一个随机向量作为后备方案  
        #     return [np.random.random() for _ in range(1024)]  
        
        # 使用ZhipuEmbedding类获取嵌入向量
        result = self.embedding_model.get_single_embedding(text).tolist()
        print("使用智谱embedding转为向量后的结果是：", result[0])
        print("embedding.length = ", len(result[0]))
        return result[0]
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:  
        """执行RAG检索和回答生成"""  
        try:  
            # 获取查询的嵌入向量  
            query_embedding = self._get_embedding(query)  
            
            # 检索相关文档  
            search_results = self.vector_db.search(query_embedding, top_k=3)  
            
            # 构建上下文  
            context_docs = [doc for doc, score in search_results]  
            context_text = "\n".join(context_docs)  
            
            # 构建提示  
            prompt = f"""你是一个专业的上海旅游顾问，请根据以下信息回答用户的问题。  
                    参考信息:  
                    {context_text}  

                    用户问题: {query}  

                    请用简洁专业的语言回答上述问题，如果参考信息中没有相关内容，你可以说"我没有足够的信息回答这个问题"。"""  
            
            # 调用智谱AI生成回答  
            response = client.chat.completions.create(  
                model="glm-4-fast",  
                messages=[  
                    {"role": "user", "content": prompt}  
                ]  
            )  
            
            answer = response.choices[0].message.content  
            
            return {  
                "agent": self.name,  
                "answer": answer,  
                "sources": context_docs,  
                "query": query  
            }  
        except Exception as e:  
            print(f"RAG执行失败: {e}")  
            return {  
                "agent": self.name,  
                "answer": f"抱歉，我在处理您的请求时遇到了问题: {str(e)}",  
                "query": query  
            }  



class SearchAgent(BaseAgent):  
    """搜索执行Agent: 负责通过SerpAPI获取上海旅游信息的执行智能体"""  
    
    def __init__(self, name: str = "搜索智能体"):  
        super().__init__(name)  
        self.api_key = SERPAPI_API_KEY  
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:  
        """执行搜索并返回结果"""  
        try:  
            # 构建搜索查询  
            search_query = f"上海旅游 {query}"  
            
            # 使用SerpAPI执行搜索  
            search_params = {  
                "q": search_query,  
                "api_key": self.api_key,  
                "engine": "google",  
                "google_domain": "google.com.hk",  
                "gl": "cn",  
                "hl": "zh-cn",  
                "num": 5  # 返回5条结果  
            }  
            
            search = GoogleSearch(search_params)  
            results = search.get_dict()  
            
            # 提取搜索结果  
            organic_results = results.get("organic_results", [])  
            
            # 构建搜索结果文本  
            if organic_results:  
                search_context = []  
                for i, result in enumerate(organic_results[:5], 1):  
                    title = result.get("title", "无标题")  
                    snippet = result.get("snippet", "无摘要")  
                    search_context.append(f"{i}. {title}: {snippet}")  
                
                search_text = "\n".join(search_context)  
                
                # 构建提示  
                prompt = f"""你是一个专业的上海旅游顾问，请根据以下搜索结果回答用户的问题。  
                    搜索结果:  
                    {search_text}  

                    用户问题: {query}  

                    请用简洁专业的语言回答上述问题，基于搜索结果提供信息。如果搜索结果中没有相关内容，请说明。"""  
                
                # 调用智谱AI生成回答  
                response = client.chat.completions.create(  
                    model="glm-4",  
                    messages=[  
                        {"role": "user", "content": prompt}  
                    ]  
                )  
                
                answer = response.choices[0].message.content  
                
                return {  
                    "agent": self.name,  
                    "answer": answer,  
                    "sources": search_context,  
                    "query": query  
                }  
            else:  
                return {  
                    "agent": self.name,  
                    "answer": "抱歉，我没有找到相关的搜索结果。",  
                    "query": query  
                }  
        except Exception as e:  
            print(f"搜索执行失败: {e}")  
            return {  
                "agent": self.name,  
                "answer": f"抱歉，我在搜索时遇到了问题: {str(e)}",  
                "query": query  
            }  


# 整合Agent  
class CoordinatorAgent(BaseAgent):  
    """协调多个执行智能体的整合智能体"""  
    
    def __init__(self, name: str = "协调智能体"):  
        super().__init__(name)  
        self.rag_agent = RAGAgent()  
        self.search_agent = SearchAgent()  
        self.conversation_history = []  
    
    def _decide_agent(self, query: str) -> str:  
        """决定使用哪个执行智能体来回答问题"""  
        # 构建提示  
        prompt = f"""作为一个决策者，你需要确定使用哪个智能体来回答以下关于上海旅游的问题。  

            问题: {query}  

            可选的智能体:  
            1. RAG智能体 - 从本地知识库中检索相关信息并回答问题  
            2. 搜索智能体 - 通过Google搜索获取最新的上海旅游信息  

            请仅回复"RAG"或"SEARCH"，选择最适合回答这个问题的智能体。如果问题是关于基本的上海旅游信息，请使用RAG智能体；如果是关于最新信息、具体景点详情、价格、评价等需要实时数据的问题，请使用搜索智能体。"""  
        
        # 调用智谱AI做决策  
        response = client.chat.completions.create(  
            model="glm-4",  
            messages=[  
                {"role": "user", "content": prompt}  
            ]  
        )  
        
        decision = response.choices[0].message.content.strip().upper()  
        
        # 规范化决策结果  
        if "RAG" in decision:  
            return "RAG"  
        else:  
            return "SEARCH"  
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:  
        """运行整合智能体，协调执行智能体回答问题"""  
        try:  
            # 记录当前查询到对话历史  
            self.conversation_history.append({"role": "user", "content": query})  
            
            # 决定使用哪个执行智能体  
            agent_type = self._decide_agent(query)  
            
            # 根据决策调用相应的执行智能体  
            if agent_type == "RAG":  
                result = self.rag_agent.run(query)  
            else:  # SEARCH  
                result = self.search_agent.run(query)  
            
            # 获取答案  
            answer = result["answer"]  
            agent_used = result["agent"]  
            
            # 构建整合响应  
            prompt = f"""你是一个专业的上海旅游顾问，请根据以下信息回答用户的问题。请使用自然、友好的语言，确保回答是有帮助的。  

                用户问题: {query}  

                {agent_used}提供的回答:  
                {answer}  

                请对上述回答进行优化，确保语言自然、信息完整、态度友好。不要提及是哪个智能体回答的问题，直接回答用户问题即可。"""  
            
            # 调用智谱AI生成最终回答  
            response = client.chat.completions.create(  
                model="glm-4",  
                messages=[  
                    {"role": "user", "content": prompt}  
                ]  
            )  
            
            final_answer = response.choices[0].message.content  
            
            # 记录回答到对话历史  
            self.conversation_history.append({"role": "assistant", "content": final_answer})  
            
            return {  
                "answer": final_answer,  
                "agent_used": agent_type,  
                "original_result": result  
            }  
        except Exception as e:  
            print(f"整合智能体执行失败: {e}")  
            error_message = f"抱歉，我在处理您的请求时遇到了问题: {str(e)}"  
            self.conversation_history.append({"role": "assistant", "content": error_message})  
            return {  
                "answer": error_message,  
                "agent_used": "ERROR",  
                "error": str(e)  
            }  


# 主交互循环  
def main():  
    """主函数，处理用户输入并展示智能体回答"""  
    print("欢迎使用上海旅游助手！请输入您的问题，输入'退出'结束对话。")  
    
    # 初始化整合智能体  
    coordinator = CoordinatorAgent()  
    
    while True:  
        # 获取用户输入  
        user_query = input("\n您的问题: ")  
        
        # 检查是否退出  
        if user_query.lower() in ['退出', 'exit', 'quit']:  
            print("感谢使用上海旅游助手，再见！")  
            break  
        
        # 处理空输入  
        if not user_query.strip():  
            print("请输入您的问题。")  
            continue  
        
        print("正在思考...")  
        
        # 运行整合智能体处理查询  
        result = coordinator.run(user_query)  
        
        # 打印回答  
        print(f"\n助手: {result['answer']}")  
        
        # 调试信息  
        if os.environ.get("DEBUG") == "true":  
            print("\n--- 调试信息 ---")  
            print(f"使用的智能体: {result['agent_used']}")  
            if 'original_result' in result and 'sources' in result['original_result']:  
                print(f"信息来源: {result['original_result']['sources']}")  
            print("---------------")  


if __name__ == "__main__":  
    '''
    python -m src.models.agent
    '''
    # main()  
    
    
    rag_agent = RAGAgent()
    
    rag_agent._get_embedding("哈哈哈")