import random
import json
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Setting device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading intents from JSON
with open('intents.json', 'r', encoding="utf8") as json_data:
    intents = json.load(json_data)

# Loading trained Torch intent classification model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Loading sentiment analysis model
model_sentiments = load_model('model_sentiment.keras')

# Loading tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# Setting max sequence length (must match training)
max_length = model_sentiments.input_shape[1]

bot_name = "BeCalmAI"

def preprocess_text(text):
    """Tokenizes and pads input for sentiment analysis"""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    return np.array(padded)

def get_response(msg):
    """Classifies user input for intent and sentiment and returns an appropriate response."""
    if not msg.strip():
        return "I'm here to help! 😊", [0]

    # Tokenization and preprocessing for intent classification
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    # Preprocessing text for sentiment analysis
    X_sentiments_padded = preprocess_text(msg)

    # Predicting intent
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Predicting sentiment
    output_sentiments = model_sentiments.predict(X_sentiments_padded)
    sentiment_score = np.argmax(output_sentiments)

    # Debugging - Print model predictions
    print(f"Sentiment Model Output: {output_sentiments}")
    print(f"Predicted Sentiment Index: {sentiment_score}")

    # Sentiment mapping
    sentiment_labels = {0: "joy", 1: "sadness", 2: "support", 3: "funny", 4: "thanks", 5: "goodbye"}
    sentiment = sentiment_labels.get(sentiment_score, "neutral")

    # Generating response
    response = "I'm here to help! 😊"
    for intent in intents['intents']:
        if tag == intent['tag']:
            base_response = random.choice(intent['responses'])
            if sentiment == "joy":
                response = f"{base_response} I'm glad you're happy! 😊"
            elif sentiment == "sadness":
                response = f"{base_response} I'm here for you. You're not alone. 💙"
            elif sentiment == "support":
                response = f"{base_response} You got this! Keep going. 💪"
            elif sentiment == "funny":
                response = f"{base_response} Haha, that's funny! 😆"
            elif sentiment == "thanks":
                response = f"{base_response} You're very welcome! 😊"
            elif sentiment == "goodbye":
                response = f"{base_response} Goodbye! Take care! 👋"
            else:
                response = base_response

    return response, [sentiment_score]

if __name__ == "__main__":
    print(f"{bot_name}: Hello! How can I assist you today? (Type 'quit' to exit)")

    sentiment_scores = []

    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            print(f"{bot_name}: Goodbye! Have a great day! 😊")
            break

        answer, score = get_response(sentence)
        sentiment_scores.append(score[0])

        print(f"{bot_name}: {answer}")
        print("Sentiment Scores:", sentiment_scores)
