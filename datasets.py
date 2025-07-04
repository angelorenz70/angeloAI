import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

class CustomDatasets:
    def __init__(self, dataset_path="datasets/datasets.pkl", tokenizer_path="angeloai_tokenizer.pkl", remove_stopwords=False):
        self.dataset_path = dataset_path
        self.tokenizer_path = tokenizer_path
        self.remove_stopwords = remove_stopwords

        # Load datasets
        self.load_dataset()
        self.remove_duplicates()

        # Lazy initialized data
        self._processed_data = None
        self._tokenizer = None
        self._input_sequences = None
        self._max_seq_length = None

    def load_dataset(self):
        with open(self.dataset_path, "rb") as f:
            self.datasets = pickle.load(f)
        print(f"📄 Loaded {len(self.datasets)} records from dataset.")

    def remove_duplicates(self):
        print("🔎 Checking for duplicates...")
        initial_count = len(self.datasets)
        self.datasets = list(dict.fromkeys(self.datasets))
        removed_count = initial_count - len(self.datasets)
        print(f"🗑️ Removed {removed_count} duplicate entries." if removed_count else "✅ No duplicates found.")

    def concat_datasets(self):
        return [s.split() for s in self.datasets]

    def preprocess(self, data):
        processed = []
        stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()
        for sentence in data:
            words = [word.lower() for word in sentence if word.isalpha()]
            if self.remove_stopwords:
                words = [word for word in words if word not in stop_words]
            processed.append(words)
        return processed

    def get_processed_data(self):
        if self._processed_data is None:
            self._processed_data = self.preprocess(self.concat_datasets())
            print(f"✅ Preprocessed {len(self._processed_data)} sentences.")
        return self._processed_data

    def get_tokenizer(self):
        if self._tokenizer is None:
            try:
                with open(self.tokenizer_path, "rb") as f:
                    self._tokenizer = pickle.load(f)
                print("✅ Tokenizer loaded from file.")
            except FileNotFoundError:
                print("⚠️ No saved tokenizer found. Fitting new tokenizer...")
                self._tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
                self._tokenizer.fit_on_texts(self.get_processed_data())
                with open(self.tokenizer_path, "wb") as f:
                    pickle.dump(self._tokenizer, f)
                print(f"✅ Tokenizer saved to {self.tokenizer_path}.")
        return self._tokenizer

    def generate_ngrams(self, shuffle=False):
        if self._input_sequences is None:
            sequences = self.get_tokenizer().texts_to_sequences(self.get_processed_data())
            self._input_sequences = []
            for seq in sequences:
                for i in range(1, len(seq)):
                    self._input_sequences.append(seq[:i + 1])
            if shuffle:
                np.random.shuffle(self._input_sequences)
                print("🔀 Shuffled n-grams.")
            print(f"📈 Generated {len(self._input_sequences)} n-grams.")
        return self._input_sequences

    def pad_sequences(self):
        ngrams = self.generate_ngrams()
        self._max_seq_length = max(len(seq) for seq in ngrams)
        print(f"📏 Max sequence length: {self._max_seq_length}")
        return pad_sequences(ngrams, maxlen=self._max_seq_length, padding='pre')

    def split_labels(self):
        sequences = np.array(self.pad_sequences())
        X = sequences[:, :-1]
        y = sequences[:, -1]
        print(f"🔑 Split into {len(X)} samples (X) and {len(y)} labels (y).")
        return X, y

    def get_vocab_size(self):
        tokenizer = self.get_tokenizer()
        vocab_size = min(len(tokenizer.word_index) + 1, 5000)
        print(f"🔤 Vocabulary size: {vocab_size}")
        return vocab_size

    def get_max_seq_length(self):
        if self._max_seq_length is None:
            _ = self.pad_sequences()  # Triggers computation
        return self._max_seq_length
