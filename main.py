from chatbot import Chatbot

# bot = Chatbot()
# encoded = bot.encode_prompt("Hello, how are you?")
# print(encoded)
# reply = bot.decode_reply(encoded['input_ids'][0].tolist())
# print(reply)
# prompt = "What is the weather like today?"
# reply = bot.generate_reply(prompt)
# print(f"Prompt: {prompt}")
# print(f"Reply: {reply}")

def main():
    bot = Chatbot()

    # Print a welcome message
    print("=== Welcome to Chatbot ===")
    print("System prompt:", bot.system_prompt.strip())
    print("Type 'exit' or 'quit' to end the chat.\n")

    # Conversation loop
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        reply = bot.generate_reply(user_input)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()