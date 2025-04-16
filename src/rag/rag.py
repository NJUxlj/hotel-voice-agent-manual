import os  
import json  
import numpy as np  
from typing import List, Dict, Any, Optional, Literal, Tuple
import torch  
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM  
from sentence_transformers import SentenceTransformer  
import faiss  
from tqdm import tqdm  

class FAQRAG:  
    """  
    基于FAQ问答对的检索增强生成(RAG)系统  
    """  
    
    def __init__(  
        self,  
        embedding_type: Literal["zhipu", "huggingface"] = "huggingface",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  
        agent = None, 
        device: str = None,  
        index_type: str = "l2",  
        top_k: int = 3  
    ):  
        """  
        初始化FAQ RAG系统  
        
        参数:  
            embedding_model_name: 用于生成嵌入的模型名称  
            llm_model_name: 用于生成回答的语言模型名称  
            device: 运行设备 ('cuda', 'cpu', 或None自动选择)  
            index_type: FAISS索引类型 ('l2', 'ip', 'cosine')  
            top_k: 检索时返回的最相似结果数量  
        """  
        # 确定设备  
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  
        self.top_k = top_k  
        
        self.embedding_type = embedding_type
        
        self.embedding_model_name = embedding_model_name
        
        print(f"初始化嵌入模型: {embedding_model_name}")  
        self.embedding_model = SentenceTransformer(embedding_model_name)  
        self.embedding_model.to(self.device)  
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()  
        
        # 创建FAISS索引  
        if index_type == 'cosine':     # 余弦相似度 = 向量A·向量B / (||A|| * ||B||, 当向量归一化后(长度为1)，余弦相似度 = 向量A·向量B
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积，需要归一化向量  
            self.normalize = True  
        elif index_type == 'ip':  
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积  
            self.normalize = False  
        else:  # 'l2'  
            self.index = faiss.IndexFlatL2(self.embedding_dim)  # 欧氏距离   √Σ(Ai - Bi)² 
            self.normalize = False  
            
        # 存储数据  
        self.qa_pairs = []  
        self.questions = []  
        self.answers = []  
        
 

        
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:  
        """  
        生成文本的嵌入向量  
        
        参数:  
            texts: 要生成嵌入的文本列表  
            
        返回:  
            嵌入向量数组  
        """  
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)  
        if self.normalize:  
            faiss.normalize_L2(embeddings)  
        return embeddings  
    
    def load_from_json(self, json_file: str) -> None:  
        """  
        从JSON文件加载问答对数据  
        
        参数:  
            json_file: JSON文件的路径  
        """  
        print(f"从{json_file}加载问答对数据...")  
        with open(json_file, 'r', encoding='utf-8') as f:  
            self.qa_pairs = json.load(f)  
        
        self.questions = [pair["question"] for pair in self.qa_pairs]  
        self.answers = [pair["answer"] for pair in self.qa_pairs]  
        
        print(f"加载了{len(self.qa_pairs)}个问答对")  
    
    def add_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> None:  
        """  
        添加问答对到系统  
        
        参数:  
            qa_pairs: 包含问答对的列表，每个问答对是一个包含'question'和'answer'键的字典  
        """  
        self.qa_pairs.extend(qa_pairs)  
        self.questions.extend([pair["question"] for pair in qa_pairs])  
        self.answers.extend([pair["answer"] for pair in qa_pairs])  
        
        print(f"添加了{len(qa_pairs)}个新问答对，总计{len(self.qa_pairs)}个")  
    
    def build_index(self) -> None:  
        """构建FAISS索引"""  
        if not self.questions:  
            print("警告: 没有可索引的问题，请先加载问答对数据")  
            return  
        
        print("生成问题的嵌入向量...")  
        question_embeddings = self._get_embeddings(self.questions)  
        
        print("构建向量索引...")  
        self.index.reset()  # 重置索引  
        self.index.add(question_embeddings)  # 添加向量到索引  
        
        print(f"索引构建完成，包含{self.index.ntotal}个向量")  
    
    def save_index(self, index_file: str, metadata_file: str) -> None:  
        """  
        保存FAISS索引和元数据  
        
        参数:  
            index_file: 保存索引的文件路径  
            metadata_file: 保存元数据的文件路径  
        """  
        # 保存索引  
        print(f"保存索引到{index_file}...")  
        faiss.write_index(self.index, index_file)  
        
        # 保存问答对元数据  
        print(f"保存元数据到{metadata_file}...")  
        with open(metadata_file, 'w', encoding='utf-8') as f:  
            json.dump({  
                'qa_pairs': self.qa_pairs,  
                'questions': self.questions,  
                'answers': self.answers  
            }, f, ensure_ascii=False, indent=2)  
        
        print("保存完成")  
    
    def load_index(self, index_file: str, metadata_file: str) -> None:  
        """  
        加载FAISS索引和元数据  
        
        参数:  
            index_file: 索引文件路径  
            metadata_file: 元数据文件路径  
        """  
        # 加载索引  
        print(f"从{index_file}加载索引...")  
        self.index = faiss.read_index(index_file)  
        
        # 加载元数据  
        print(f"从{metadata_file}加载元数据...")  
        with open(metadata_file, 'r', encoding='utf-8') as f:  
            metadata = json.load(f)  
            self.qa_pairs = metadata['qa_pairs']  
            self.questions = metadata['questions']  
            self.answers = metadata['answers']  
        
        print(f"加载完成，索引包含{self.index.ntotal}个向量，元数据包含{len(self.qa_pairs)}个问答对")  
    
    def retrieve(self, query: str, return_scores: bool = False) -> List[Dict[str, Any]]:  
        """  
        检索与查询最相似的问答对  
        
        参数:  
            query: 用户查询文本  
            return_scores: 是否返回相似度分数  
            
        返回:  
            检索到的问答对列表，按相似度排序  
        """  
        # 生成查询的嵌入向量  
        query_embedding = self._get_embeddings([query])  
        
        # 搜索最相似的向量  
        scores, indices = self.index.search(query_embedding, self.top_k)  
        
        # 整理结果  
        results = []  
        for i, idx in enumerate(indices[0]):  
            if idx < 0 or idx >= len(self.qa_pairs):  # 检查索引是否有效  
                continue  
                
            result = {  
                'question': self.questions[idx],  
                'answer': self.answers[idx],  
                'qa_pair': self.qa_pairs[idx]  
            }  
            
            if return_scores:  
                result['score'] = float(scores[0][i])  
                
            results.append(result)  
            
        return results  
    
    def _format_prompt(self, query: str, retrieved_results: List[Dict[str, Any]]) -> str:  
        """  
        根据检索结果格式化提示词  
        
        参数:  
            query: 用户查询  
            retrieved_results: 检索到的问答对列表  
            
        返回:  
            格式化的提示词  
        """  
        context_parts = []  
        for i, result in enumerate(retrieved_results, 1):  
            q = result['question']  
            a = result['answer']  
            context_parts.append(f"参考问题{i}: {q}\n参考答案{i}: {a}")  
        
        context = "\n\n".join(context_parts)  
        
        prompt = f"""请基于以下FAQ中检索到的参考问答，回答用户的问题。  
                    如果参考问答能够直接回答问题，请直接使用参考答案；  
                    如果参考问答不能完全回答问题，但提供了相关信息，请基于参考信息组织回答；  
                    如果参考问答与用户问题无关，请礼貌地表示无法回答。  
                    请保持回答准确、客观、简洁。  

                    参考FAQ信息：  
                    {context}  

                    用户问题: {query}  

                    回答:"""  
        
        return prompt  
    
    def generate_response(self, query: str, mode: str = "rag") -> Dict[str, Any]:  
        """  
        生成对用户查询的回答  
        
        参数:  
            query: 用户查询  
            mode: 回答模式：  
                 - 'direct': 直接返回最相似问题的答案  
                 - 'rag': 使用检索增强生成回答(需要LLM)  
            
        返回:  
            包含回答和检索结果的字典  
        """  
        # 检索相关问答对  
        retrieved_results = self.retrieve(query, return_scores=True)  
        
        if not retrieved_results:  
            return {  
                'query': query,  
                'answer': "很抱歉，我没有找到相关信息，无法回答您的问题。",  
                'retrieved': []  
            }  
        
        if mode == 'direct':  
            # 直接使用最相似问题的答案  
            top_result = retrieved_results[0]  
            return {  
                'query': query,  
                'answer': top_result['answer'],  
                'retrieved': retrieved_results  
            }  
        
        elif mode == 'rag':  
            # 检查LLM是否已初始化  
            if not self.llm_model or not self.llm_tokenizer:  
                # 没有LLM时回退到直接模式  
                print("警告: 未初始化LLM，回退到直接回答模式")  
                return self.generate_response(query, mode='direct')  
            
            # 构建提示词  
            prompt = self._format_prompt(query, retrieved_results)  
            
            # 生成回答  
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)  
            
            with torch.no_grad():  
                outputs = self.llm_model.generate(  
                    inputs.input_ids,  
                    max_new_tokens=512,  
                    temperature=0.7,  
                    top_p=0.9,  
                    repetition_penalty=1.1  
                )  
                
            response = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  
            
            return {  
                'query': query,  
                'answer': response,  
                'retrieved': retrieved_results,  
                'prompt': prompt  
            }  
        
        else:  
            raise ValueError(f"不支持的模式: {mode}")  
    
    def interactive_chat(self, mode: str = "rag") -> None:  
        """  
        启动交互式对话界面  
        
        参数:  
            mode: 回答模式 ('direct' 或 'rag')  
        """  
        print(f"=== FAQ RAG 交互式对话 (模式: {mode}) ===")  
        print("输入问题开始对话，输入'exit'或'quit'结束")  
        
        while True:  
            query = input("\n用户: ").strip()  
            
            if query.lower() in ['exit', 'quit', '退出']:  
                print("再见!")  
                break  
                
            if not query:  
                continue  
                
            print("思考中...")  
            response = self.generate_response(query, mode=mode)  
            
            print(f"\n助手: {response['answer']}")  
            
            # 打印检索结果（调试用）  
            if os.environ.get("DEBUG") == "1":  
                print("\n--- 检索结果 ---")  
                for i, result in enumerate(response['retrieved'], 1):  
                    print(f"{i}. 问题: {result['question']}")  
                    print(f"   答案: {result['answer']}")  
                    if 'score' in result:  
                        print(f"   相似度: {result['score']:.4f}")  
                    print()  


# 使用示例  
def main():  
    # 初始化RAG系统  
    rag = FAQRAG(  
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言嵌入模型，适合中文  
        llm_model_name="THUDM/chatglm3-6b",  # 使用ChatGLM-3-6B作为语言模型  
        index_type="cosine",  # 使用余弦相似度  
        top_k=3  # 检索前3个最相似的结果  
    )  
    
    # 加载问答对数据（假设已经提取并保存为JSON文件）  
    json_file = "extracted_qa.json"  
    
    # 检查文件是否存在，如果不存在则创建示例数据  
    if not os.path.exists(json_file):  
        print(f"{json_file}不存在，创建示例数据...")  
        sample_text = """1、客服中心营业时间？  
答案：开园前半小时，闭园后半小时。  
2、公园停车楼收费标准是什么？  
答案：11:30 之前入场 80 元/次，11:30 之后入场 50 元/次。30分钟内免费。  
3、门票购买以后是否可以退票退款？  
答案：园现场售卖的实体门票一经售出不退不换。网络平台上购买的电子门票，必须 是未使用状态下的可以联系第三方平台办理退票  
4、上海海昌海洋公园残疾人需要买门票吗？  
答案：需要，残疾人可持残疾证和有效身份证件购买优待票  
5、上海海昌海洋公园的现场购票付款方式有几种？  
答案：有 4 种，分别是现金，微信，支付宝，刷卡  
6、办理的公园年卡该怎么入园？  
答案：游客可在线上、线下开通年卡，在入园检票口刷脸识别入园即可，但优待年卡或 者亲子年"""  
        
        # 使用已有的函数提取问答对  
        from extract_qa import extract_qa_pairs  
        qa_pairs = extract_qa_pairs(sample_text)  
        
        # 保存为JSON文件  
        with open(json_file, 'w', encoding='utf-8') as f:  
            json.dump(qa_pairs, f, ensure_ascii=False, indent=4)  
    
    # 加载问答对数据  
    rag.load_from_json(json_file)  
    
    # 构建索引  
    rag.build_index()  
    
    # 保存索引和元数据（可选）  
    rag.save_index("faq_index.bin", "faq_metadata.json")  
    
    # 启动交互式对话  
    rag.interactive_chat(mode="rag")  # 使用RAG模式  

if __name__ == "__main__":  
    main()  