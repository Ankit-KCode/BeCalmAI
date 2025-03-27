import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from sklearn.model_selection import train_test_split
from collections import Counter

# Load JSON file
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Define sentiment mapping
sentiment_mapping = {
    "joy": 0, "sad": 1, "support": 2, "funny": 3, "thanks": 4, "goodbye": 5
}

# Extract patterns and labels
sentiment_data = [
    (pattern, sentiment_mapping[intent["tag"]]) 
    for intent in data.get("intents", []) if intent["tag"] in sentiment_mapping 
    for pattern in intent.get("patterns", [])
]

# Ensure data is available
if not sentiment_data:
    raise ValueError("No sentiment-related data found in intents.json!")

# Separate texts and labels
texts, labels = zip(*sentiment_data)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")  
tokenizer.fit_on_texts(texts)

# Debugging tokenization
vocab_size = len(tokenizer.word_index) + 1
print(f"Tokenizer vocab size: {vocab_size}")  

# Convert texts to sequences and pad
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)  
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to NumPy array
labels = np.array(labels)

# Check class distribution
label_counts = Counter(labels)
print("Class distribution:", label_counts)

# Ensure test size is at least equal to the number of classes
min_test_size = max(len(label_counts), 2)  # At least 2 test samples
test_size = max(0.2, min_test_size / len(labels))  # Dynamic test size

# Use stratify only if each class has at least 2 samples
use_stratify = all(count >= 2 for count in label_counts.values())

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences, labels, test_size=test_size, random_state=42,
    stratify=labels if use_stratify else None
)

print(f"Train size: {len(X_train)}, Test size: {len(X_val)}")  # Debugging

# Define LSTM model
embedding_dim = 32  
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=False)),  
    Dropout(0.4),  
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(sentiment_mapping), activation='softmax')  
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model with validation data
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=8, verbose=1)  

# Save model
model.save("model_sentiment.keras")

# Save tokenizer
with open("tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer.to_json(), f)

print("Sentiment model trained and saved as 'model_sentiment.keras'")
