from chatbot import Chatbot

bot = Chatbot()
encoded = bot.encode_prompt("Hello, how are you?")
print(encoded)
reply = bot.decode_reply(encoded['input_ids'][0].tolist())
print(reply)
prompt = "What is the weather like today?"
reply = bot.generate_reply(prompt)
print(f"Prompt: {prompt}")
print(f"Reply: {reply}")