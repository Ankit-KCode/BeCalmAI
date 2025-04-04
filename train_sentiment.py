import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import RandomOverSampler  # For balancing dataset

# Loading JSON file
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Defining sentiment mapping
sentiment_mapping = {
    "joy": 0, "sadness": 1, "support": 2, "funny": 3, "thanks": 4, "goodbye": 5
}

# Extracting patterns and labels
sentiment_data = []
for intent in data.get("intents", []):
    tag = intent.get("tag")
    if tag in sentiment_mapping:
        for pattern in intent.get("patterns", []):
            sentiment_data.append((pattern, sentiment_mapping[tag]))

if not sentiment_data:
    raise ValueError("No sentiment-related data found in intents.json!")

# Separating texts and labels
texts, labels = zip(*sentiment_data)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Converting texts to sequences and pad them
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)  # Finding max sequence length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Converting labels to NumPy array
labels = np.array(labels)

# Checking class distribution
label_counts = Counter(labels)
print("Class distribution before balancing:", label_counts)

# Balance dataset using Oversampling
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = ros.fit_resample(padded_sequences, labels)

print("Class distribution after balancing:", Counter(y_resampled))

# Spliting dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"Train size: {len(X_train)}, Test size: {len(X_val)}")

# Defining LSTM Model with Dropout
embedding_dim = 32
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_length),
    Dropout(0.2),  # Added dropout to prevent overfitting
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(sentiment_mapping), activation='softmax')  
])

# Compilling Model with RMSprop optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Adding Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training Model
model.fit(X_train, y_train, validation_data=(X_val, y_val), 
          epochs=20, batch_size=8, verbose=1, callbacks=[early_stopping])

# Saving Model
model.save("model_sentiment.keras")

# Saving Tokenizer
with open("tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer.to_json(), f)

print("✅ Sentiment model trained and saved as 'model_sentiment.keras'")
