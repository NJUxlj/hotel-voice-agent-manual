import os  
import re  
import json  
import docx  
from docx.table import Table  
from typing import List, Dict, Any, Optional, Tuple

import docx.text
import docx.text.paragraph  

import soundfile as sf  
from datasets import load_dataset  
import random 

from src.configs.config import (
    SHANGHAI_TOURISM_DOCX,
    CANTONESE_DATA_PATH
)


def is_question(text: str) -> bool:  
    """判断一段是否为问题"""  
    return bool(re.match(r'^(\d+)[、\.．]{1}', text)) or bool(re.match(r'^\d+', text))  



def extract_table_content(table: Table) -> str:  
    """  
    提取Word表格内容并转换为格式化文本  
    
    参数:  
        table: docx文档中的表格对象  
        
    返回:  
        格式化后的表格文本  
    """  
    table_content = []  
    
    # 获取表头（如果存在）  
    headers = []  
    if len(table.rows) > 0:  
        for cell in table.rows[0].cells:  
            headers.append(cell.text.strip())  
    
    # 处理表格内容  
    for i, row in enumerate(table.rows):  
        # 跳过表头行  
        if i == 0 and any(h for h in headers if h):  # 如果第一行有非空表头，认为是表头  
            continue  
            
        row_content = []  
        for j, cell in enumerate(row.cells):  
            cell_text = cell.text.strip()  
            if cell_text:  
                # 如果有表头，添加表头作为前缀  
                if j < len(headers) and headers[j]:  
                    row_content.append(f"{headers[j]}: {cell_text}")  
                else:  
                    row_content.append(cell_text)  
        
        # 只添加非空行  
        if row_content:  
            table_content.append(" | ".join(row_content))  
    
    # 返回格式化后的表格内容  
    return "\n".join(table_content)  


def extract_qa_from_docx(docx_path: str) -> List[Dict[str, str]]:  
    """  
    从Word文档(.docx)中提取问答对  
    
    参数:  
        docx_path: Word文档路径  
        
    返回:  
        问答对列表  
    """  
    # 加载docx文档  
    doc = docx.Document(docx_path)  
    
    qa_pairs = []  
    current_question = None  
    current_answer = []  
    in_table_answer = False  
    
    
    def save_qa():  
        nonlocal current_question, current_answer  
        if current_question and current_answer:  
            question = re.sub(r'^(\d+)[、\.．]?', '', current_question).strip()  
            answer_text = "\n".join([l for l in current_answer if l.strip()])  
            qa_pairs.append({  
                'question': question,  
                'answer': answer_text  
            })  
        current_question = ''  
        current_answer = []  
    
    # 遍历文档中的所有元素  
    for element in doc.element.body:  
        # 处理段落  
        if element.tag.endswith('p'):  
            paragraph = docx.text.paragraph.Paragraph(element, doc)  
            text = paragraph.text.strip()  
            
            # 如果是空段落，跳过  
            if not text:  
                continue  
            
            # 检查是否是问题（匹配数字+、+文本）  
            if is_question(text):  
                # 如果已经有问题和答案，保存之前的问答对  
                if current_question and (current_answer or in_table_answer):  
                    # 清理问题文本（移除序号）  
                    clean_question = re.sub(r'^(\d+)[、\.．]{1}', '', current_question).strip()  
                    
                    # 合并答案文本  
                    answer_text = "\n".join(current_answer).strip()  
                    
                    # 添加到问答对列表  
                    qa_pairs.append({  
                        "question": clean_question,  
                        "answer": answer_text  
                    })  
                
                # 设置当前问题  
                current_question = text  
                current_answer = []  
                in_table_answer = False  
            
            # 检查是否是"答案："格式的答案  
            elif text.startswith('答案：') or (current_question and not re.match(r'^(\d+)[、\.．]{1}', text)) or ( current_question and re.match(r'^[①-⑩]', text)):  
                # 提取答案内容  
                answer_text = text
                if answer_text.startswith('答案：'):
                    answer_text = text[3:].strip()  # 移除"答案："前缀  
                current_answer.append(answer_text.strip())  
            
            # # 否则，如果已有问题，认为是答案的一部分  
            # elif current_question:  
            #     current_answer.append(text)  
        
        # 处理表格  
        elif element.tag.endswith('tbl'):  
            table = docx.table.Table(element, doc)  
            
            # 如果当前有问题，认为表格是答案的一部分  
            if current_question:  
                table_text = extract_table_content(table)  
                if table_text:  
                    current_answer.append(table_text)  
                    in_table_answer = True  
    
    # 处理最后一个问答对  
    if current_question and (current_answer or in_table_answer):  
        clean_question = re.sub(r'^(\d+)[、\.．]{1}', '', current_question).strip()  
        answer_text = "\n".join(current_answer).strip()  
        
        qa_pairs.append({  
            "question": clean_question,  
            "answer": answer_text  
        })  
    
    return qa_pairs  


def process_docx_to_json(docx_path: str, output_json: str = None) -> List[Dict[str, str]]:  
    """  
    处理Word文档并将提取的问答对保存为JSON  
    
    参数:  
        docx_path: Word文档路径  
        output_json: 输出JSON文件路径，如果不提供则只返回结果不保存  
        
    返回:  
        问答对列表  
    """  
    try:  
        # 检查文件是否存在  
        if not os.path.exists(docx_path):  
            print(f"错误: 文件 '{docx_path}' 不存在")  
            return []  
        
        # 提取问答对  
        qa_pairs = extract_qa_from_docx(docx_path)  
        
        # 打印提取结果统计  
        print(f"从 '{docx_path}' 中提取了 {len(qa_pairs)} 个问答对")  
        
        # 如果提供了输出路径，保存为JSON  
        if output_json:  
            with open(output_json, 'w', encoding='utf-8') as f:  
                json.dump(qa_pairs, f, ensure_ascii=False, indent=4)  
            print(f"问答对已保存到 '{output_json}'")  
        
        return qa_pairs  
        
    except Exception as e:  
        print(f"处理失败: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        return []  


def cleanup_answer(answer: str) -> str:  
    """  
    清理答案文本，去除多余空白和格式化问题  
    
    参数:  
        answer: 原始答案文本  
        
    返回:  
        清理后的答案文本  
    """  
    # 替换多个空格为单个空格  
    answer = re.sub(r'\s+', ' ', answer)  
    
    # 处理表格内容中的格式  
    lines = answer.split('\n')  
    cleaned_lines = []  
    
    for line in lines:  
        # 清理每一行  
        line = line.strip()  
        if line:  
            cleaned_lines.append(line)  
    
    # 重新组合  
    return '\n'.join(cleaned_lines)  




def post_process_qa_pairs(qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:  
    """  
    对提取的问答对进行后处理，优化格式和内容  
    
    参数:  
        qa_pairs: 原始问答对列表  
        
    返回:  
        处理后的问答对列表  
    """  
    processed_pairs = []  
    
    for pair in qa_pairs:  
        # 清理问题  
        question = pair["question"].strip()  
        
        # 清理答案  
        answer = cleanup_answer(pair["answer"])  
        
        # 添加到处理后的列表  
        processed_pairs.append({  
            "question": question,  
            "answer": answer  
        })  
    
    return processed_pairs  





class QaDataGenerator:
    def __init__(self, docx_path: str):
        
        self.docx_path = docx_path
        self.qa_pairs = []
        self.current_index = 0
        
        self._load()
        
    def _load(self):
        self.qa_pairs = process_docx_to_json(self.docx_path)
        self.current_index = 0
        print(f"从 '{self.docx_path}' 中加载了 {len(self.qa_pairs)} 个问答对")

    def get_next_qa_pair(self) -> Tuple[str, str]:
        if self.current_index >= len(self.qa_pairs):
            self.current_index = 0
            print("已循环所有问答对，重新开始")
            self._load()  # 重新加载

        qa_pair = self.qa_pairs[self.current_index]
        self.current_index += 1
        return qa_pair["question"], qa_pair["answer"]
    
    def get_all_qa_pairs(self) -> List[Dict[str, str]]:
        return self.qa_pairs
    
    def __len__(self):
        return len(self.qa_pairs)
    
    
    def __getitem__(self, idx):
        return self.qa_pairs[idx]





class CantoneseVoiceDataGenerator:
    '''
    加载数据时有报错，暂时无法解决。
    '''
    
    def __init__(self, data_path, subset_name = "yue"):
        self.data_path = data_path  # safecantonese/cantomap
        self.dataset = load_dataset(data_path, "yue", split="train") 
        # self.dataset = load_dataset("mozilla-foundation/common_voice_16_0", "yue", trust_remote_code=True)  
        print(f"数据集大小: {len(self.dataset)} 条音频")  
    
        self.output_dir = "src/data/cantonese_audio_samples"  
        os.makedirs(self.output_dir, exist_ok=True) 
        
    
    def sample_k_data(self, k=5):
        # 随机选择几个样本并保存为WAV文件  
        num_samples = k 
        random_indices = random.sample(range(len(self.dataset)), num_samples)  

        for i, idx in enumerate(random_indices):  
            sample = self.dataset[idx]  
            
            # 获取音频数据  
            audio_array = sample["audio"]["array"]  
            sampling_rate = sample["audio"]["sampling_rate"]  
            
            # 保存为WAV文件  
            output_path = os.path.join(self.output_dir, f"cantonese_sample_{i+1}.wav")  
            sf.write(output_path, audio_array, sampling_rate)  
            
            # 输出样本信息  
            print(f"已保存样本 {i+1} 到 {self.output_path}")  
            print(f"对应的文本: {sample['sentence']}")  
            print(f"采样率: {sampling_rate} Hz")  
            print(f"音频长度: {len(audio_array)/sampling_rate:.2f} 秒")  
            print("-" * 50)  

        print(f"共保存了 {num_samples} 个音频样本到 {self.output_dir} 目录") 

    

if __name__ == "__main__":  
    
    '''
    python -m src.data.load
    '''

    # generator = QaDataGenerator(SHANGHAI_TOURISM_DOCX)
    
    
    # print(generator.get_next_qa_pair())
    
    dg = CantoneseVoiceDataGenerator(CANTONESE_DATA_PATH+"/audio")
    
    dg.sample_k_data()
    
