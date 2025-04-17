# hotel-voice-agent-manual







## 项目流程

1. 语音输入： 用户可以使用粤语语音提问。

2. ASR： 将粤语语音转换为中文文本。可以使用开源模型 (例如：WeNet) 或商用 API (例如：阿里云语音识别)。

3. 知识库：中文知识库，见附件。

4. RAG： 使用 RAG (Retrieval-Augmented Generation) 方案，从知识库中检索相关信息。

5. LLM： 使用 LLM (Large Language Model) 根据检索到的信息生成中文答案。可以使用开源模型 (例如：ChatGLM) 或商用 API (例如：OpenAI GPT)。当然如果直接生成粤语答案更佳。

6. TTS： 将中文答案转换为粤语语音。可以使用开源模型 (例如：Mitts) 或商用 API (例如：腾讯云语音合成)。

7. 语音输出： 使用粤语语音回答用户提出的问题。


## 运行结果
- 全部放在根目录的 `image` 文件夹下。





## 项目配置

### 更新langchain
```
pip install -U langchain
```

### ZhipuAIEmbedding依赖下载
```
pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install fastembed
```

### 环境变量设置
- `ZHIPU_API_KEY`
```
export ZHIPU_API_KEY="您的实际API密钥"
```


- `DASHSCOPE_API_KEY`  [Tongyi API Key]
```
pip install dashscope
export DASHSCOPE_API_KEY="your_api_key_here"

# 如何使用：

def __init__(self, model_type="tongyi", dashscope_api_key=None):
    if model_type == "tongyi":
        # 直接传入 API 密钥
        self.model = Tongyi(dashscope_api_key="your_api_key_here")
    # ... 其他代码 ...
```
- `SERPAPI_API_KEY` # SerpAPI API Key
```
export SERPAPI_API_KEY="您的实际API密钥"
```

- 或者修改全局变量：
```
vim ~/.bashrc
# 在文件末尾添加以下内容
export ZHIPU_API_KEY="您的实际API密钥"
export DASHSCOPE_API_KEY="your_api_key_here"
export SERPAPI_API_KEY="您的实际API密钥"
# 保存并退出文件
source ~/.bashrc
```

### 下载Docx插件
```
pip install python-docx
```