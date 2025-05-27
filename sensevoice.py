from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import sounddevice as sd
import numpy as np
import torch
import time

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",    
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# 音频录制参数
sample_rate = 16000  # 16kHz采样率
duration = 2  # 录制10秒，可根据需要调整
channels = 1  # 单声道

print("开始录音...请说话...")
# 录制音频
audio_data = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=channels,
    dtype='float32'
)
sd.wait()  # 等待录制完成
print("录音结束")
start = time.time()
# 将音频数据转换为模型需要的格式
audio_np = np.squeeze(audio_data)  # 去除多余的维度
print(audio_np)

audio_tensor = torch.from_numpy(audio_np).float()  # 转换为PyTorch张量

# en
res = model.generate(
    input=audio_tensor,
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)

elapsed = time.time() - start
print(f"处理时间: {elapsed:.2f}s")
