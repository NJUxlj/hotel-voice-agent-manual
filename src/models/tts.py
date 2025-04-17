import os  
import json  
import time  
import requests  
import pyaudio  
import wave  
import dashscope  
from dashscope.audio.asr import *  
from tencentcloud.common import credential  
from tencentcloud.common.profile.client_profile import ClientProfile  
from tencentcloud.common.profile.http_profile import HttpProfile  
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException  
from tencentcloud.tts.v20190823 import tts_client, models  

TENCENT_SECRET_ID = os.environ.get("TENCENT_SECRET_ID")  # 腾讯云SecretId  
TENCENT_SECRET_KEY = os.environ.get("TENCENT_SECRET_KEY")   # 腾讯云SecretKey  

# 腾讯云TTS处理粤语语音合成  
class TencentTTS:  
    def __init__(self):  
        self.cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)  
        hp = HttpProfile()  
        hp.endpoint = "tts.tencentcloudapi.com"  
        cp = ClientProfile()  
        cp.httpProfile = hp  
        self.client = tts_client.TtsClient(self.cred, "ap-guangzhou", cp)  
    
    def synthesize_speech(self, text, output_file="output.mp3"):  
        """将文本合成为语音"""  
        try:  
            # 创建腾讯云TTS请求参数  
            req = models.TextToVoiceRequest()  
            req.Text = text  
            req.SessionId = str(int(time.time()))  
            req.Volume = 5  # 音量，范围：[-10,10]，默认为0  
            req.Speed = 0   # 语速，范围：[-2,6]，默认为0  
            req.VoiceType = 101006  # 粤语女声：101006，粤语男声：101007  
            req.PrimaryLanguage = 1  # 中文  
            req.SampleRate = 16000  # 采样率  
            req.Codec = "mp3"  # 音频格式  
            req.ModelType = 1  # 默认值
            
            # 调用API  
            response = self.client.TextToVoice(req)  
            
            # 将Base64编码的音频内容保存为文件  
            audio_content = response.Audio  
            import base64  
            with open(output_file, "wb") as f:  
                f.write(base64.b64decode(audio_content))  
            
            print(f"语音合成完成，已保存到: {output_file}")  
            return output_file  
        
        except TencentCloudSDKException as err:  
            print(f"腾讯云TTS错误: {err}")  
            return None  
        
        
        
        
if __name__ == "__main__":
    '''
    python -m src.models.tts
    
    '''
    
    tts = TencentTTS()
    
    tts.synthesize_speech("你今天也很棒哦！！")
    
    