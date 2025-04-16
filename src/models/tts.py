from openai import OpenAI

client = OpenAI(api_key="your_key")

def text_to_speech(text):
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",  # 适合酒店场景的柔和女声
        input=text,
        speed=0.9      # 优化语速提升自然度
    )
    response.stream_to_file("output.mp3")