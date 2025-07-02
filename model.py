from tensorflow.keras.models import load_model
import pickle

with open('angeloai_dataset.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print(tokenizer)


