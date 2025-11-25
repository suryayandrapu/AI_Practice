import ollama

# Stream the response (useful for long answers)
stream = ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Write a short story about AI'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
