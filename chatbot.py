from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    def __init__(self):
        # Use a compact conversational model
        model_name = "microsoft/DialoGPT-small"

        # Expose tokenizer and model for later use
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Added in task 6
        self.chat_history_ids = None

    # Added in task 6
    def reset_history(self):
        """Clear stored conversation history."""
        self.chat_history_ids = None    

    def encode_prompt(self, prompt: str):
        # Convert an input string into model-friendly tensors (PyTorch)
        return self.tokenizer(prompt, return_tensors="pt")
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        # Convert token IDs back into a human-readable string
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    

    # TASK 5 GENERATE_REPLY
    # def generate_reply(self, prompt: str) -> str:
    #     """Generate a natural-language reply from the model for a single prompt."""
    #     # Some models behave better with a newline at the end of the prompt
    #     prompt_with_nl = prompt + "\n"

    #     # 1) Encode the prompt
    #     encoded = self.encode_prompt(prompt_with_nl)
    #     input_len = encoded["input_ids"].shape[-1]

    #     # 2) Generate output token IDs
    #     output_ids = self.model.generate(
    #         **encoded,
    #         max_new_tokens=128,
    #         pad_token_id=self.tokenizer.eos_token_id,
    #         do_sample=True,
    #         temperature=0.9,
    #         top_p=0.8,
    #         top_k=50,
    #     )

    #     # 3) Slice out only the newly generated tokens (after the prompt)
    #     new_token_ids = output_ids[0, input_len:].tolist()

    #     # 4) Decode to text and return
    #     reply_text = self.decode_reply(new_token_ids).strip()
    #     return reply_text

    # TASK 6 GENERATE_REPLY
    def generate_reply(self, prompt: str) -> str:
        """Generate a reply while preserving conversation context."""
        # 1) Encode the new user prompt (newline can help some models)
        prompt_with_nl = prompt + "\n"
        encoded = self.encode_prompt(prompt_with_nl)

        # 2) Build model input by concatenating with prior history if present
        if self.chat_history_ids is not None:
            model_input_ids = torch.cat([self.chat_history_ids, encoded["input_ids"]], dim=-1)
            attention_mask = torch.cat([torch.ones_like(self.chat_history_ids), encoded["attention_mask"]], dim=-1)
        else:
            model_input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        # 3) Generate continuation tokens
        output_ids = self.model.generate(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            top_k=50,
        )

        # 4) Extract only the newly generated portion (after the input length)
        input_len = model_input_ids.shape[-1]
        new_token_ids = output_ids[0, input_len:].tolist()

        # 5) Append the new output to the running history for next turn
        self.chat_history_ids = output_ids

        # 6) Decode and return the reply text
        return self.decode_reply(new_token_ids).strip()