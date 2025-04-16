# hotel-voice-agent-manual







## 项目流程

1. 语音输入： 用户可以使用粤语语音提问。

2. ASR： 将粤语语音转换为中文文本。可以使用开源模型 (例如：WeNet) 或商用 API (例如：阿里云语音识别)。

3. 知识库：中文知识库，见附件。

4. RAG： 使用 RAG (Retrieval-Augmented Generation) 方案，从知识库中检索相关信息。

5. LLM： 使用 LLM (Large Language Model) 根据检索到的信息生成中文答案。可以使用开源模型 (例如：ChatGLM) 或商用 API (例如：OpenAI GPT)。当然如果直接生成粤语答案更佳。

6. TTS： 将中文答案转换为粤语语音。可以使用开源模型 (例如：Mitts) 或商用 API (例如：腾讯云语音合成)。

7. 语音输出： 使用粤语语音回答用户提出的问题。
