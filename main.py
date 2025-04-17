from src.models.pipline import VoiceChatPipline

from src.data.load import CantoneseVoiceDataGenerator

from src.configs.config import CANTONESE_DATA_PATH

def generate_cantonese_samples():
    dg = CantoneseVoiceDataGenerator(CANTONESE_DATA_PATH)

    dg.sample_k_data(k=5)



def test_pipline():
    file_path = "src/data/录音.wav"
    pipline = VoiceChatPipline(file_path=file_path)  # 你也可以不用广东话数据集，自己传一段音频进去
    pipline.run()
    
    
    
    


if __name__ == '__main__':
    test_pipline()