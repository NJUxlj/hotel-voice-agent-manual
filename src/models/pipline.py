# 把从 ASR->LLM->RAG->TTS 的整个流程串起来

import os
import json
import random

from src.models.agent import CoordinatorAgent
from src.models.asr import AliyunASR
from src.models.tts import TencentTTS


WAV_FILE_PATH = "src/data/录音.wav"


class VoiceChatPipline:
    
    
    def __init__(self, file_path=WAV_FILE_PATH, use_microphone=False):
        self.use_microphone = use_microphone
        self.file_path = file_path
        self.asr=AliyunASR()
        
        self.tts = TencentTTS()
        
        
        self.agent = CoordinatorAgent()
        
        
    def run(self, truncate_length = 200):
        if self.use_microphone:
            result = self.asr.recognize_from_microphone()
        else:
            result = self.asr.recognize_from_file(self.file_path)

        result = self.agent.run(result)
        
        result = result['answer'][:200]
        print("RAG输出：",result)
        print("由于腾讯云的限制，RAG的输出回答被手动截断到200~~")
        
        output_file = "src/data/output.mp3"
        self.tts.synthesize_speech(result, output_file = output_file)
        
        print(f"语音合成成功，被保存在本地目录：{output_file}")



    
    
    
    




if __name__ == "__main__":
    '''
    python -m src.models.pipline
    '''
    pipline = VoiceChatPipline(file_path="src/data/粤语录音.wav")
    pipline.run()
    