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
from datetime import datetime

import numpy as np
import pickle
import matplotlib.pyplot as plt


class AngeloAIModel:
    def __init__(self, vocab_size, max_seq_length, X, y, tokenizer):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.model = None
        self.history = None
        self.now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=200, input_length=self.max_seq_length - 1))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dense(self.vocab_size, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def summary(self):
        self.model.summary()

    def train(self, epochs=200, batch_size=32, model_name_prefix='angeloai_model'):
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f"models/{model_name_prefix}{self.now}-best.h5", save_best_only=True)

        self.history = self.model.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint]
        )

        # Save final model
        final_model_path = f"models/{model_name_prefix}_{self.now}.h5"
        self.model.save(final_model_path)
        print(f"📁 Model saved to {final_model_path}")

        # Save tokenizer
        with open("angeloai_tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)
        print("🧠 Tokenizer saved as angeloai_tokenizer.pkl")

    def plot_accuracy(self):
        if self.history:
            plt.plot(self.history.history['accuracy'])
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.show()
        else:
            print("No training history to plot.")

    def sample_with_temperature(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

    def generate_text(self, seed_text, next_words=5, temperature=0.7):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_seq_length - 1, padding='pre')
            predicted = self.model.predict(token_list, verbose=0)
            predicted_word_index = self.sample_with_temperature(predicted[0], temperature)
            predicted_word = self.tokenizer.index_word.get(predicted_word_index, '')
            seed_text += ' ' + predicted_word
        return seed_text
