from ollama import chat

print("Chat started! Type 'that's all' to end the conversation.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["that's all", "thats all", "that's all.", "exit", "quit", "bye"]:
        print("Goodbye! Have a great day!")
        break
    
    response = chat(model='lumen:latest', messages=[
        {
            'role': 'user',
            'content': user_input,
        },
    ])
    
    print(f"\nAssistant: {response['message']['content']}\n")