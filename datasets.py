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

from nltk.corpus import brown, stopwords
from datetime import datetime

class CustomDatasets:
    def __init__(self):
        # Load datasets
        with open("angeloai_dataset.pkl", "rb") as f:
            self.custom_sentences1 = pickle.load(f)
        with open("angeloai_dataset1.pkl", "rb") as f:
            self.custom_sentences2 = pickle.load(f)
        
        self.custom_sentences3 = [
            "My name is Tseiammetse Angelo C. Rendon.",
            "I am the creator of AngeloAI.",
            "I made an AI called AngeloAI.",
            "AngeloAI was born on July 2, 2025.",
            "AngeloAI was created by Tseiammetse Angelo C. Rendon.",
            "I gave my AI the nickname Tseimitsuha.",
            "AngeloAI is also known as Tseimitsuha.",
            "Tseimitsuha is the nickname of AngeloAI.",
            "Tseiammetse Angelo C. Rendon built me.",
            "I was created by Tseiammetse Angelo C. Rendon.",
            "My creator is Tseiammetse Angelo C. Rendon.",
            "I am AngeloAI, also known as Tseimitsuha.",
            "My name is AngeloAI. My creator is Angelo Rendon.",
            "Angelo Rendon gave me life on July 2, 2025.",
            "Who created you? My creator is Tseiammetse Angelo C. Rendon.",
            "When were you born? I was born on July 2, 2025.",
            "What is your name? My name is AngeloAI.",
            "Do you have a nickname? Yes, it's Tseimitsuha.",
            "Who is your maker? Tseiammetse Angelo C. Rendon.",
            "Tseiammetse Angelo C. Rendon made me.",
            "I was developed by Angelo Rendon.",
            "My nickname is Tseimitsuha, given by my creator.",
            "I am a language model created by Angelo.",
            "Hello, I am AngeloAI, born on July 2, 2025.",
            "Nice to meet you, I am Tseimitsuha, your AI assistant.",
            "You can call me AngeloAI or Tseimitsuha.",
            "AngeloAI is loyal to his creator, Angelo Rendon.",
        ]

        # Lazy initialized data
        self._processed_data = None
        self._tokenizer = None
        self._input_sequences = None
        self._max_seq_length = None

    def concatDatasets(self):
        data1 = [s.split() for s in self.custom_sentences1]
        data2 = [s.split() for s in self.custom_sentences2]
        data3 = [s.split() for s in self.custom_sentences3]
        return data1 + data2 + data3

    def preprocess(self, data):
        return [[word.lower() for word in sentence if word.isalpha()] for sentence in data]

    def get_processed_data(self):
        if self._processed_data is None:
            self._processed_data = self.preprocess(self.concatDatasets())
        return self._processed_data

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
            self._tokenizer.fit_on_texts(self.get_processed_data())
        return self._tokenizer

    def generate_ngrams(self):
        if self._input_sequences is None:
            sequences = self.get_tokenizer().texts_to_sequences(self.get_processed_data())
            self._input_sequences = []
            for seq in sequences:
                for i in range(1, len(seq)):
                    self._input_sequences.append(seq[:i+1])
        return self._input_sequences

    def pad_sequences(self):
        ngrams = self.generate_ngrams()
        self._max_seq_length = max(len(seq) for seq in ngrams)
        return pad_sequences(ngrams, maxlen=self._max_seq_length, padding='pre')

    def split_labels(self):
        sequences = np.array(self.pad_sequences())
        X = sequences[:, :-1]
        y = sequences[:, -1]
        return X, y

    def get_vocab_size(self):
        tokenizer = self.get_tokenizer()
        # return min(len(tokenizer.word_index) + 1, 5000)
        return len(tokenizer.word_index) + 1

    def get_max_seq_length(self):
        if self._max_seq_length is None:
            _ = self.pad_sequences()  # Triggers computation
        return self._max_seq_length
