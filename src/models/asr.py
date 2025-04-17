import os  
import json  
import time  
import dashscope.audio
import requests  
import pyaudio  
import wave  
import dashscope  
from dashscope.audio.asr import *  


from pydub import AudioSegment   # 音频格式转换



# from tencentcloud.common import credential  
# from tencentcloud.common.profile.client_profile import ClientProfile  
# from tencentcloud.common.profile.http_profile import HttpProfile  
# from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException  
# from tencentcloud.tts.v20190823 import tts_client, models  



ALIYUN_API_KEY = os.environ.get("DASHSCOPE_API_KEY")  # 阿里云API密钥  

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")  # 智谱API密钥（如果使用）  
LOCAL_MODEL_PATH = None  # 本地模型路径（如果使用）



# 配置API密钥  
dashscope.api_key = ALIYUN_API_KEY  




# 阿里云ASR处理粤语语音识别  
class AliyunASR:  
    def __init__(self):  
        self.callback = ASRCallback()  
        
    def recognize_from_microphone(self, duration=5)->str:  
        """从麦克风录制并识别语音"""  
        # 录制音频  
        audio_data = self._record_audio(duration)  
        
        # 保存为临时WAV文件  
        temp_file = "temp_input.wav"  
        with wave.open(temp_file, 'wb') as wf:  
            wf.setnchannels(1)  
            wf.setsampwidth(2)  
            wf.setframerate(16000)  
            wf.writeframes(audio_data)  
        
        # 识别音频文件  
        return self.recognize_from_file(temp_file)  
    
    def recognize_from_file(self, file_path)->str:  
        """识别语音文件"""  
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        # 添加API密钥验证检查
        if not dashscope.api_key:
            raise ValueError("阿里云API密钥未设置或无效")
        
        temp_file = "temp_converted.wav"
        try:
            sound:AudioSegment = AudioSegment.from_file(file_path)
            sound = sound.set_channels(1).set_frame_rate(16000)
            sound.export(temp_file, format="wav")
            file_path = temp_file  # 使用转换后的文件
            
            with wave.open(file_path, 'rb') as wf:
                print(f"音频文件参数 - 声道数: {wf.getnchannels()}, 采样宽度: {wf.getsampwidth()}, 采样率: {wf.getframerate()}")
        except Exception as e:
            print(f"音频文件格式检查失败: {e}")
            raise
        
        print(f"正在识别wav格式的语音文件: {file_path}")  
        
        # 使用阿里云的Gummy模型进行语音识别，支持粤语  
        recognition = TranslationRecognizerRealtime(    # 不要用 Recognition类
            model='gummy-realtime-v1',  # 支持粤语的模型  
            format='wav',  
            sample_rate=16000,  
            language_hints=['yue', 'zh'],  # 指定语言为粤语和普通话  
            callback=self.callback,
            transcription_enabled=True,  
            translation_enabled=False,  # 设置为False，如果不需要翻译  
        )  
        
        recognition.start()  
        
        try:  
            with open(file_path, 'rb') as f:  
                # 读取文件并按块发送  
                while True:  
                    audio_data = f.read(12800)  
                    if not audio_data:  
                        break  
                    print(f"发送音频数据块，大小: {len(audio_data)}字节")
                    recognition.send_audio_frame(audio_data)  
                    # dashscope.audio.asr.Transcription.wait()
                    
            # 等待识别完成  
            # time.sleep(5)  
            start_time = time.time()
            while not self.callback.final_text and time.time() - start_time < 15:
                time.sleep(0.5)
            
        except Exception as e:
            print(f"音频识别过程中发生错误: {e}")
            raise
        finally:  
            recognition.stop()  
        
        # 返回识别结果  
        result = self.callback.get_final_result()  
    
        if not result:
                print("警告: 未获取到音频识别的结果")
                print("可能原因:")
                print("1. 音频文件格式不符合要求(需要16kHz, 16bit, 单声道WAV)")
                print("2. 阿里云API密钥无效")
                print("3. 网络连接问题")
                print("4. 音频内容无法识别")
        return result
    
    def _record_audio(self, duration):  
        """录制音频"""  
        print(f"开始录音，请说话（{duration}秒）...")  
        
        CHUNK = 1024  
        FORMAT = pyaudio.paInt16  
        CHANNELS = 1  
        RATE = 16000  
        
        p = pyaudio.PyAudio()  
        stream = p.open(format=FORMAT,  
                        channels=CHANNELS,  
                        rate=RATE,  
                        input=True,  
                        frames_per_buffer=CHUNK)  
        
        frames = []  
        for i in range(0, int(RATE / CHUNK * duration)):  
            data = stream.read(CHUNK)  
            frames.append(data)  
        
        print("录音结束")  
        stream.stop_stream()  
        stream.close()  
        p.terminate()  
        
        return b''.join(frames)  

# 阿里云ASR回调处理  
class ASRCallback(RecognitionCallback):  
    def __init__(self):  
        self.final_text = ""  
        
    def on_open(self):  
        print("ASR会话已打开")  
        
    def on_close(self):  
        print("ASR会话已关闭")  
        
    # def on_event(self, result: RecognitionResult)->None:  
    #     print(f"收到事件")  
    #     print("result = ", result)
    #     sentence = result.get_sentence()  
    #     print("sentence = ", sentence)
    #     if sentence and 'text' in sentence:  
    #         print(f"识别中: {sentence['text']}")  
    #         # 如果是最终结果，更新final_text  
    #         if RecognitionResult.is_sentence_end(sentence):  
    #             self.final_text = sentence['text'] 
    
    def on_event(self, request_id, transcription_result, translation_result, usage) -> None:  
        print(f"收到事件, request_id: {request_id}")  
        
        if transcription_result:  
            print(f"转录结果类型: {type(transcription_result)}")  
            print(f"转录结果内容: {transcription_result}")  
        
        if transcription_result and hasattr(transcription_result, 'sentences'):  
            for sentence in transcription_result.sentences:  
                if hasattr(sentence, 'text') and sentence.text:  
                    print(f"识别中({sentence.status}): {sentence.text}")  
                    if sentence.status == "FINAL":  
                        self.final_text = sentence.text  
                        print(f"[最终结果] {self.final_text}") 
                        
        elif hasattr(transcription_result, 'text') and transcription_result.text:  
                print(f"识别结果: {transcription_result.text}")  
                self.final_text = transcription_result.text  
                print(f"[最终结果] {self.final_text}")  
                
        elif isinstance(transcription_result, dict) or hasattr(transcription_result, '__getitem__'):  
                try:  
                    if 'text' in transcription_result:  
                        print(f"识别结果: {transcription_result['text']}")  
                        self.final_text = transcription_result['text']  
                        print(f"[最终结果] {self.final_text}")  
                    elif 'sentences' in transcription_result:  
                        for sentence in transcription_result['sentences']:  
                            if 'text' in sentence:  
                                print(f"识别中({sentence.get('status', 'UNKNOWN')}): {sentence['text']}")  
                                if sentence.get('status') == 'FINAL':  
                                    self.final_text = sentence['text']  
                                    print(f"[最终结果] {self.final_text}")  
                except Exception as e:  
                    print(f"尝试字典访问失败: {e}")  
    
    def on_error(self, result)->None:  
        print(f"ASR错误: {result}")  
        
    def on_complete(self) -> None:  
        print("ASR识别完成")  
        
    def get_final_result(self):  
        return self.final_text
    
    
    



if __name__ == '__main__':
    '''
    python -m src.models.asr
    '''
    asr = AliyunASR()
    
    
    
    result = asr.recognize_from_file("src/data/录音.wav")

    print(result)