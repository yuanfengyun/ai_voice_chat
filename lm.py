import requests
import json

from openai import OpenAI
import re

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
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response.append(content)
    str = "".join(full_response)
    result = re.sub(r'<think>.*?</think>', '', str, flags=re.DOTALL)
    return result.replace('\n', '').replace('\r', '')

def main_fun():
    messages = [
            {"role": "system", "content": "你是一个有用的助手"}
    ]
    text = "你好，你是谁？"
    print("转录结果:", text)
    messages.append({"role": "user", "content": text})
    res = call_llm(messages)
	messages.append({"role": "system", "content": res})
    print("deepseek返回:", res)

main_fun()