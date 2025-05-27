import whisper
import sounddevice as sd
import numpy as np
import time
import torch

import shutil
import os

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = whisper.load_model("medium").cuda()

# 录音参数
fs = 16000  # 采样率
duration = 5  # 录音时长(秒)

print("开始录音...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
sd.wait()  # 等待录音完成
print("结束录音...")
# 保存为numpy数组并转录
audio = recording.flatten()
start = time.time()
result = model.transcribe(audio)
elapsed = time.time() - start
print("转录结果:", result["text"])
print(f"音频时长: {result['segments'][-1]['end']:.2f}s")
print(f"处理时间: {elapsed:.2f}s")
print(f"实时率: {elapsed/result['segments'][-1]['end']:.1f}x")