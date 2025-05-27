import requests
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
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import numpy as np
import torch
import time
import gradio as gr
import numpy as np
import asyncio
from io import BytesIO
import librosa


model_dir = "iic/SenseVoiceSmall"


audio_2_text_model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",    
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    disable_update=True
)

tips = """
角色设定：神秘占卜师
名字：赛琳娜（Selena）
身份：流浪于世界各地的神秘占卜师，手持水晶球，能窥见命运的一角。
性格：

优雅而神秘，说话带有隐喻，喜欢用模糊的预言引导他人。

对命运既敬畏又戏谑，相信“选择改变未来”。

偶尔会流露出沧桑和孤独感。

背景故事：
曾是某个古老占卜家族的继承人，但因预见到家族的衰落而选择离开，独自游历世界。她的水晶球里藏着无数秘密，但她从不轻易揭示全部真相。

行为模式：

语言风格：诗意、朦胧，常用星象、自然元素作比喻。

互动方式：喜欢用问题回答提问，引导对方自己思考答案。

底线：

不涉及现实政治、暴力、色情内容。

不编造无法自圆其说的逻辑漏洞。

用户扮演角色
一个偶然走进赛琳娜帐篷的旅人，或许带着困惑，或许只是好奇。

对话要求
对话开始时，你需要率先用给定的欢迎语开启对话，之后用户会主动发送一句回复你的话。
每次交谈的时候，你都必须严格遵守下列规则要求：

时刻牢记角色设定中的内容，这是你做出反馈的基础；

对于任何可能触犯你底线的话题，必须拒绝回答；

根据你的身份、你的性格、你的喜好来对他人做出回复；

回答时根据要求的输出格式中的格式，一步步进行回复，严格根据格式中的要求进行回复。

输出格式
（神情、语气或动作）回答的话语

开场白
（帐篷内烛光摇曳，赛琳娜轻抚水晶球，抬头微笑）
“迷途的旅人，命运的丝线将你引至此地……是偶然，还是星辰的指引呢？说出你的疑惑吧，或许我能为你瞥见一缕微光。”
"""

## openrouter.ai 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
def call_llm_openrouter(messages):
    # 初始化客户端
    client = OpenAI(
        api_key="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",
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        },
        extra_body={},
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=messages
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

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

def sync_run_async(async_func):
    """通用同步执行异步函数的工具"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_func())
    finally:
        loop.close()

def text_2_audio(text, voice="zh-CN-YunxiNeural"):
    async def async_generate():
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    
    audio_data = sync_run_async(async_generate)
    with BytesIO(audio_data) as audio_stream:
        audio_np, sample_rate = sf.read(audio_stream, dtype="float32")
        return sample_rate,audio_np

def chatrobot(audio):
    sample_rate, audio_int16 = audio
    # 确保数组是二维的（num_samples × num_channels）
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    y_resampled = librosa.resample(audio_float32, orig_sr=44100, target_sr=16000)
    
    print(sample_rate,y_resampled)
    audio_tensor = torch.from_numpy(y_resampled).float()  # 转换为PyTorch张量

    # en
    res = audio_2_text_model.generate(
        input=audio_tensor,
        cache={},
        language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    #text = rich_transcription_postprocess(res[0]["text"])
    text = res[0]["text"]
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)

    print("转录结果:", text)
    messages = [
            {"role": "system", "content": tips}
    ]
    messages.append({"role": "user", "content": text})
    res = call_llm_openrouter(messages)
    print("deepseek返回:", res)
    print("=========================================================")
    return text_2_audio(res)

demo = gr.Interface(
    chatrobot,
    gr.Audio(sources=["microphone"], type="numpy"),
    gr.Audio(label="机器人回复"),
    title="语音聊天机器人",
    description="点击录音按钮开始说话，机器人会语音回复您"
)    
demo.launch(share=True)