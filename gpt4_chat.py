import openai
import sys
import json
from time import sleep

model_name = sys.argv[1]

with open("api_keys.json") as f:
    config = json.load(f)

openai.api_type = config[model_name]["type"]
openai.api_base = config[model_name]["base"]
openai.api_version = config[model_name]["version"]
openai.api_key = config[model_name]["key"]


# 定义一个函数来与GPT-3.5进行对话
def chat_with_gpt3(prompt, m):
    response = openai.ChatCompletion.create(
        engine="gpt-4-32k",
        messages=m,
        max_tokens=1024,  # 限制生成的回复长度
        temperature=0.7  # 控制回复的创造性
    )
    return response['choices'][0]['message']['content']

messages = [
    {
        'role': 'system',
        'content': 'You are a helpful and precise assistant for researchers.'
    }
]

# 与GPT-3.5进行对话
while True:
    user_input = input("你: ")
    if user_input.lower() == '退出':
        break
    messages.append({
        "role": "user",
        "content": user_input
    })
    for i in range(20):
        try:
            response = chat_with_gpt3(user_input, messages)
            break
        except Exception as e:
            print(e)
            sleep(5)
            continue
    messages.append({
        "role": "system",
        "content": response
    })
    print("AI: " + response)
