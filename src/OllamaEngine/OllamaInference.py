# ollama inference.py
from ollama import chat

messages = []

def inference(text):
    messages.append({'role': 'user', 'content': text})
    stream = chat(
        model='llama3.2:3b',
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        piece = chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
        messages.append({'role': 'assistant', 'content': piece})