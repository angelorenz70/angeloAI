import tensorflow as tf
import nltk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from datasets import CustomDatasets
from model import AngeloAIModel
from nltk.corpus import brown, stopwords
from datetime import datetime

model_name = 'angeloai_model2025_07_03_01_57_53-best.h5'
model = load_model(f'models/{model_name}')
with open('angeloai_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
max_seq_length = 10

class AngeloAIInferenceWrapper(AngeloAIModel):
    def __init__(self, model, tokenizer, max_seq_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

angelo_model = AngeloAIInferenceWrapper(model, tokenizer, max_seq_length)

# 🌀 Input Loop
print("🤖 AngeloAI is ready. Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == 'exit':
        print("AngeloAI: Goodbye! 👋")
        break
    response = angelo_model.generate_text(user_input, next_words=10, temperature=0.9).strip().capitalize()
    print(f"AngeloAI: {response}")
