from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import json
import os

class chatbt():
    def __init__(self):
        model_name = "microsoft/DialoGPT-medium"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.chat_history_file = 'chat_history.json'
        if not os.path.exists(self.chat_history_file):
            with open(self.chat_history_file, 'w') as file:
                json.dump({"history":[]}, file)

    def load_chat_history(self):
        with open(self.chat_history_file, 'r') as file:
            return json.load(file).get('history', [])

    def save_chat_history(self, user_input, bot_resp):
        with open(self.chat_history_file, 'r+') as file:
            data = json.load(file)
            data['history'].append({"User":user_input, "Bot":bot_resp})
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

    def generate_responce(self, user_input):
        chat_history = self.load_chat_history()
        #chat_history.append(user_input)
        chat_history_str = ''.join([f"User:{item['User']} Bot:{item['Bot']}" for item in chat_history])
        chat_history_ids = self.tokenizer.encode(chat_history_str + self.tokenizer.eos_token, return_tensors = 'pt')
        self.user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        self.bot_input_ids = torch.cat([chat_history_ids, self.user_input_ids], dim=-1)
        attn_mask = torch.ones(self.bot_input_ids.shape, dtype=torch.long)
        with torch.no_grad():
            self.chat_history_ids = self.model.generate(self.bot_input_ids, max_length = 500, pad_token_id = self.tokenizer.eos_token_id, attention_mask = attn_mask)
        bot_responce = self.tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1] :][0], skip_special_tokens = True)
        chat_history.append(bot_responce)
        return bot_responce

    def clear_chat_history(self):
        if os.path.exists(self.chat_history_file):
            os.remove(self.chat_history_file)

def main():
    st.title("Chatbot")
    chatbot = chatbt()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    chat_container = st.container()
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])           
    if user_input := st.chat_input("Say something..."):
        with st.chat_message("User"):
            st.markdown(user_input)
            st.session_state.chat_history.append({"role":"User", "content":user_input})
        bot_responce = chatbot.generate_responce(user_input)
        with st.chat_message("Bot"):
            st.markdown(bot_responce)
            st.session_state.chat_history.append({"role":"Bot", "content":bot_responce})
        chatbot.save_chat_history(user_input, bot_responce)
           
        
if __name__ == '__main__':
    main()  
