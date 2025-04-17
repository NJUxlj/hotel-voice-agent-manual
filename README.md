# hotel-voice-agent-manual



## 注意：所有运行结果都放在`image`文件夹中

- 音频输入文件放在 `src/data/录音.wav`
- 音频输出文件放在 `src/data/output.mp3`

## 项目流程

1. 语音输入： 用户可以使用粤语语音提问。

2. ASR： 将粤语语音转换为中文文本。可以使用开源模型 (例如：WeNet) 或商用 API (例如：阿里云语音识别)。

3. 知识库：中文知识库，见附件。

4. RAG： 使用 RAG (Retrieval-Augmented Generation) 方案，从知识库中检索相关信息。

5. LLM： 使用 LLM (Large Language Model) 根据检索到的信息生成中文答案。可以使用开源模型 (例如：ChatGLM) 或商用 API (例如：OpenAI GPT)。当然如果直接生成粤语答案更佳。

6. TTS： 将中文答案转换为粤语语音。可以使用开源模型 (例如：Mitts) 或商用 API (例如：腾讯云语音合成)。

7. 语音输出： 使用粤语语音回答用户提出的问题。


## 如何运行
```
python main.py
```

### 如果想测试某个具体模块
```
# 以ASR功能为例
python -m src.models.asr
```

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


### 语音数据处理
```
pip install pyaudio

# 或者
conda install -c conda-forge pyaudio

## 如果遇到依赖冲突（头文件不存在）：
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio
```


### 粤语数据集下载
```
# safecantonese/cantomap

cd src/data
huggingface-cli download --resume-download --repo-type dataset safecantonese/cantomap --local-dir cantomap
```


### 配置aliyun-ASR环境
- API-KEY：自行前往阿里云白炼平台获取
- aliyun-asr 地址： https://www.aliyun.com/sswb/887628.html
- 必须开通 ASR服务
```
vim ~/.bashrc
export DASHSCOPE_API_KEY="your_api_key_here"
source ~/.bashrc
```

### 配置腾讯云TTS环境
- 地址：https://cloud.tencent.com/product/tts
- api管理：https://console.cloud.tencent.com/cam/capi
```
pip install tencentcloud-sdk-python

# 可选 【项目中用的是aliyun的】
pip install tencentcloud-sdk-python-tts

vim ~/.bashrc
export TENCENT_SECRET_ID="your_tencent_secret_id"  # 腾讯云SecretId  
export TENCENT_SECRET_KEY="your_tencent_secret_key"  # 腾讯云SecretKey 
source ~/.bashrc
```