from aip import AipSpeech

# 百度API配置
APP_ID = 'your_app_id'
API_KEY = 'your_api_key'
SECRET_KEY = 'your_secret_key'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def cantonese_asr(audio_path):
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    result = client.asr(audio_data, 'wav', 16000, {'dev_pid': 1637})  # 1637为粤语识别
    return result.get('result')[0] if result['err_no'] == 0 else None