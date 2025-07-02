import tensorflow as tf
import nltk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

from nltk.corpus import brown, stopwords

# Download required NLTK data
with open("angeloai_dataset.pkl", "rb") as f:
    custom_sentences = pickle.load(f)


nltk.download('brown')
nltk.download('stopwords')

# Load and preprocess data
# data = brown.sents()[:5000]  # LIMIT for memory efficiency
data = [sentence.split() for sentence in custom_sentences]

# def preprocess(data):
#     stop_words = set(stopwords.words('english'))
#     processed_data = []
#     for sentence in data:
#         sentence = [word.lower() for word in sentence if word.isalpha()]
#         sentence = [word for word in sentence if word not in stop_words]
#         processed_data.append(sentence)
#     return processed_data
def preprocess(data):
    return [[word.lower() for word in sentence if word.isalpha()] for sentence in data]


processed_data = preprocess(data)

# Tokenize
# tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')  # limit vocabulary
# tokenizer.fit_on_texts(processed_data)
# sequences = tokenizer.texts_to_sequences(processed_data)
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

# Generate n-gram sequences
input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
# max_seq_length = max([len(seq) for seq in input_sequences])
# input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')
max_seq_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')

# Split input and labels
input_sequences = np.array(input_sequences)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]  # Integer labels


# Define model
vocab_size = min(len(tokenizer.word_index) + 1, 10000)  # Ensure same as tokenizer
# model = Sequential()
# model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_seq_length - 1))
# model.add(LSTM(128))
# model.add(Dense(vocab_size, activation='softmax'))
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_seq_length - 1))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

# Compile with sparse categorical crossentropy
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model (use fewer epochs to test)
# model.fit(X, y, epochs=10, batch_size=128)
# history = model.fit(X, y, epochs=50, batch_size=128, verbose=1)
history = model.fit(X, y, epochs=100, batch_size=32)
model.save('angeloai_model.h5')


import pickle
with open('angeloai_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_length - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        # predicted_word_index = np.argmax(predicted, axis=1)[0]
        predicted_word_index = sample_with_temperature(predicted[0], temperature=0.7)
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        seed_text += " " + predicted_word
    return seed_text

print(generate_text("who created you", 5))

model = load_model('angeloai_model.h5')
with open('angeloai_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# 🌀 Input Loop
print("🤖 AngeloAI is ready. Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == 'exit':
        print("AngeloAI: Goodbye! 👋")
        break
    response = generate_text(user_input, 5).strip()
    response = response.capitalize()
    print(f"AngeloAI: {response}")