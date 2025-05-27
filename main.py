import requests
import whisper
import sounddevice as sd
import numpy as np
import time
import torch
import shutil
import os
import ChatTTS
import torch
import torchaudio
from pydub import AudioSegment
from pydub.playback import play
import edge_tts
from openai import OpenAI
import re
import soundfile as sf
import sounddevice as sd




chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)


 
def call_llm(messages):
    # 初始化客户端
    client = OpenAI(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
    )
 
    stream = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-14b",
        messages=messages,
        # 超参数
        temperature=0.7, # 控制输出的随机性，值越低输出越确定
        top_p=0.9, # 控制输出的多样性，值越低输出越集中
        max_tokens=5120, # 控制生成的最大token数量
        frequency_penalty=0.5, # 减少重复内容的生成
        presence_penalty=0.5, # 鼓励模型引入新内容
        stream=True # 启用流式输出
    )
 
    full_response = []
    print("AI助手: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # 流式输出
            full_response.append(content)
    str = "".join(full_response)
    result = re.sub(r'<think>.*?</think>', '', str, flags=re.DOTALL)
    return result.replace('\n', '').replace('\r', '')

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = whisper.load_model("medium").cuda()

# 录音参数
fs = 16000  # 采样率
duration = 5  # 录音时长(秒)

def record_audio():
    print("开始录音...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # 等待录音完成
    print("结束录音...")
    # 保存为numpy数组并转录
    audio = recording.flatten()
    start = time.time()
    result = model.transcribe(audio)
    elapsed = time.time() - start
    return result["text"]

def text_2_audio(text):
    voice = edge_tts.Communicate(text=text, voice="zh-CN-YunxiNeural")  # 微软中文语音
    voice.save_sync("word_level_output.wav")
    data, samplerate = sf.read('word_level_output.wav')
    sd.play(data, samplerate)
    sd.wait()  # 等待播放完成

def main_fun():
    messages = [
            {"role": "system", "content": "你是一个有用的助手"}
    ]
    while True:
        text = record_audio()
        print("转录结果:", text)
        messages.append({"role": "user", "content": text})
        res = call_llm(messages)
        print("deepseek返回:", res)
        print("=========================================================")
        text_2_audio(res)
    
main_fun()