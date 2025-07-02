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


dataset = CustomDatasets()
X, y = dataset.split_labels()
vocab_size = dataset.get_vocab_size()
max_seq_length = dataset.get_max_seq_length()


# Train model
angelo_model = AngeloAIModel(vocab_size, max_seq_length, X, y, dataset._tokenizer)
angelo_model.summary()
angelo_model.train()

# Plot training accuracy
angelo_model.plot_accuracy()