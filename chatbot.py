from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Chatbot:
    def __init__(self):
        # Load a small pretrained conversational model from Hugging Face
        model_name = "microsoft/DialoGPT-small"

        # TASK 9: Detect if GPU (CUDA) is available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Tokenizer: turns text into token IDs (numbers the model understands)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model: takes token IDs and generates new token IDs (a reply)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Store the conversation history (as token IDs) across turns
        # We'll keep appending each new prompt and reply so the model has context.
        self.chat_history_ids = None

        # TASK 8: System prompt: a fixed instruction to guide the model's behavior
        self.system_prompt = "You are a helpful assistant. Respond to the end of this conversation accordingly.\n"


    def reset_history(self):
        """Clear the stored conversation history so the bot forgets past turns."""
        self.chat_history_ids = None

    def encode_prompt(self, prompt: str):
        """Convert a string into model-friendly tensors (PyTorch)."""
        # TASK 9: .to(self.device) moves the tensors to the GPU if available
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def decode_reply(self, reply_ids: list[int]) -> str:
        """Convert a list of token IDs back into readable text."""
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)

    # -------------------------------
    # TASK 5: single-turn reply (no memory)
    # -------------------------------
    # def generate_reply(self, prompt: str) -> str:
    #     """Generate a natural-language reply from the model for a single prompt (no memory)."""
    #     # Some models behave better with a newline at the end of the prompt
    #     prompt_with_nl = prompt + "\n"
    #
    #     # 1) Encode the prompt
    #     encoded = self.encode_prompt(prompt_with_nl)
    #     input_len = encoded["input_ids"].shape[-1]
    #
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
    #
    #     # 3) Slice out only the newly generated tokens (after the prompt)
    #     new_token_ids = output_ids[0, input_len:].tolist()
    #
    #     # 4) Decode to text and return
    #     reply_text = self.decode_reply(new_token_ids).strip()
    #     return reply_text

    # -------------------------------
    # TASK 6: multi-turn reply (with memory)
    # -------------------------------
    def generate_reply(self, prompt: str) -> str:
        """Generate a reply while preserving conversation context."""
        # Add a newline (some small dialogue models separate turns better with it)
        prompt_with_nl = prompt + "\n"

        # TASK 8: If this is the first turn, we need to add the system prompt
        if self.chat_history_ids is None:
            prompt_with_nl = self.system_prompt + prompt_with_nl

        # 1) Encode the prompt into token IDs (+ attention mask)
        encoded = self.encode_prompt(prompt_with_nl)

        # 2) If we already have conversation history, concatenate it with the new input
        if self.chat_history_ids is not None:
            # Concatenate the new input with the existing chat history, dim=-1 means add to the tensor like an append not create a new nested tensor
            model_input_ids = torch.cat([self.chat_history_ids, encoded["input_ids"]], dim=-1)
            # Build a matching attention mask: 1s for all prior tokens + 1s for new input
            attention_mask = torch.cat(
                [torch.ones_like(self.chat_history_ids), encoded["attention_mask"]],
                dim=-1,
            )
        else:
            # First turn: just use the encoded prompt
            model_input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        # TASK 9: Ensure the tensors are on the correct device
        # Not needed in this case, but good practice to ensure tensors are on the right device
        # Move the input tensors to the correct device (GPU or CPU)
        model_input_ids = model_input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device) 
        

        # 3) Ask the model to generate a continuation (the reply tokens)
        output_ids = self.model.generate(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,                 # maximum reply length (tokens)
            pad_token_id=self.tokenizer.eos_token_id,  # avoids EOS/pad warnings
            do_sample=True,                     # enables randomness (vs greedy)
            temperature=0.9,                    # higher = more random
            top_p=0.8,                          # nucleus sampling
            top_k=50,                           # sample only from top 50 tokens
        )

        # 4) Keep only the freshly generated part (exclude the prompt/history)
        input_len = model_input_ids.shape[-1]
        new_token_ids = output_ids[0, input_len:].tolist()

        # 5) Update conversation history with everything up to now
        self.chat_history_ids = output_ids

        # 6) Decode the new tokens back to text and return
        return self.decode_reply(new_token_ids).strip()