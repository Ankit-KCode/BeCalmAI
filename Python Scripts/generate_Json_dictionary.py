import json
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Load intents.json
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract words from patterns and lemmatize them
words = set()
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        for word in pattern.lower().split():
            words.add(lemmatizer.lemmatize(word))  # Lemmatize and add to set

# Create dictionary with words assigned a unique index
lemma_dict = {word: i + 1 for i, word in enumerate(sorted(words))}

# Save to dictionary_lemma.json
with open("dictionary_lemma.json", "w", encoding="utf-8") as f:
    json.dump(lemma_dict, f, indent=4, ensure_ascii=False)

print("dictionary_lemma.json generated successfully!")
