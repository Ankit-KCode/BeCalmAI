# import random
# import json

# import torch
# import numpy as np

# from model import NeuralNet
# from nltk_utils import bag_of_words, codificacion_sentence, stopwords, tokenize, padding
# from tensorflow.keras.models import load_model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('intents.json', 'r', encoding="utf8") as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# # Cargar modelos

# # Modelo torch: clasifica el tema de conversación
# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# # Modelo keras: clasifica por sentimiento 
# # Cerca de 0 joy
# # Cerca de 1 sadness
# model_sentiments = load_model('model_sentiment_lemma.h5')

# bot_name = "BeCalmAI"

# def get_response(msg, score):
#     # Limpieza de la frase para el modelo Torch
#     sentence = tokenize(msg)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     # Limpieza de la frase para el modelo Keras
#     sentence_sentiments = stopwords(sentence)
#     X_sentimientos = codificacion_sentence(sentence_sentiments)
    
#     # Predicción Torch
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
    
#     tag = tags[predicted.item()]


#     # Predict sentiment (Keras model)
#     output_sentimientos = model_sentiments.predict(padding(X_sentimientos))
#     sentiment_score = np.argmax(output_sentimientos)  # Get sentiment label index

#     # Map sentiment index to labels
#     sentiment_labels = {0: "joy", 1: "sadness", 2: "support", 3: "funny", 4: "thanks", 5: "goodbye"}
#     sentiment = sentiment_labels.get(sentiment_score, "neutral")

#     # Modify response based on sentiment
#     response = "I'm here to help! 😊"
#     for intent in intents['intents']:
#         if tag == intent['tag']:
#             if sentiment == "joy":
#                 response = random.choice(intent['responses']) + " I'm glad you're happy! 😊"
#             elif sentiment == "sadness":
#                 response = random.choice(intent['responses']) + " I'm here for you. You're not alone. 💙"
#             elif sentiment == "support":
#                 response = random.choice(intent['responses']) + " You got this! Keep going. 💪"
#             elif sentiment == "funny":
#                 response = random.choice(intent['responses']) + " Haha, that's funny! 😆"
#             elif sentiment == "thanks":
#                 response = random.choice(intent['responses']) + " You're very welcome! 😊"
#             elif sentiment == "goodbye":
#                 response = random.choice(intent['responses']) + " Goodbye! Take care! 👋"
#             else:
#                 response = random.choice(intent['responses'])

#     return response, [sentiment_score]


# if __name__ == "__main__":
#     # Print a random greeting
#     greeting_responses = [intent['responses'] for intent in intents['intents'] if intent['tag'] == 'greeting']
#     if greeting_responses:
#         print(random.choice(greeting_responses[0]))
    
#     score_resp = []
    
#     while True:
#         sentence = input("You: ")
#         if sentence.lower() == "quit":
#             break

#         answer = get_response(sentence, score_resp)
#         resp = answer[0]  # Bot response
#         score_resp.append(answer[1][0])  # Save sentiment score
        
#         print(resp)
#         print("Sentiment Scores:", score_resp)




























# # from tags word it is modified -------
#     # Predicción Keras
#     # Valor entre 0 y 1
# #     output_sentimientos = model_sentiments.predict(padding(X_sentimientos))
# #     input_score = output_sentimientos[0]
# #     #If the model can`t recognize the input, set the value to 0
# #     non_valid_input_score = [0]

# #     probs = torch.softmax(output, dim=1)
# #     prob = probs[0][predicted.item()]
# #     pred_item = prob.item()
    
# #     #Change tags in order to lead the conversation
# #     if len(score) == 1:
# #         tag = 'interest'
# #     elif len(score) == 2:
# #         tag = 'time'
# #     elif len(score) == 3:
# #         tag = 'support'
# #     #Release the direction of the speech, so user can access more conversation options or finish it anytime
# #     elif len(score) >= 4 and pred_item < 0.75 or len(score) >=4 and tag == 'about':
# #         #Tag-about prevents bot to redirect conversation to greeting phase if 'day' is used
# #         if tag == 'about':
# #             input_score = [0]
# #         score.pop(0)
# #         average = np.average(score)
# #         print(average)
# #         if average < 0.2:
# #             tag = 'joy'
# #         else:
# #             tag = 'support'

# #     #Select an answer for our bot
# #     if prob.item() > 0.75:
# #         # Return a tuple (response + user input input_score)
# #         for intent in intents['intents']:
# #             if tag == intent['tag']:
# #                 return random.choice(intent['responses']), input_score

# #     return "Sorry, what do you mean by that? \nI'm here to help", non_valid_input_score


# # if __name__ == "__main__":
# #     print(random.choice(intents['intents']['tag'=='greeting']['responses']))
# #     #print("Hi! How are you feeling right now? (type 'quit' to exit)")
# #     # Declare an array to keep track on user reponses score
# #     score_resp = []
    
# #     while True:
# #         # sentence = "I feel sad"
# #         sentence = input("You: ")
# #         if sentence == "quit":
# #             break

# #         answer = get_response(sentence, score_resp)
# #         #Bot response will be the first part of the tuple provided by the 'get_response' function
# #         resp = answer[0]
# #         #Add the score to a previous array
# #         score_resp.append(answer[1][0])
        
# #         print(resp)
# #         print(score_resp)


import random
import json
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, codificacion_sentence, stopwords, tokenize, padding
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON
with open('intents.json', 'r', encoding="utf8") as json_data:
    intents = json.load(json_data)

# Load trained Torch model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load classification model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Load sentiment analysis model
model_sentiments = load_model('model_sentiment.keras')

# Load tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

bot_name = "BeCalmAI"

def get_response(msg):
    """Classifies user input for intent and sentiment and returns an appropriate response."""
    if not msg.strip():
        return "I'm here to help! 😊", [0]

    # Tokenization and preprocessing for intent classification
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    # Tokenization for sentiment analysis
    sentence_sentiments = stopwords(sentence)
    seq = tokenizer.texts_to_sequences([sentence_sentiments])
    X_sentimientos_padded = padding(seq)
    X_sentimientos_padded = np.array(X_sentimientos_padded).reshape(1, -1)

    # Predict intent
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Predict sentiment
    output_sentimientos = model_sentiments.predict(X_sentimientos_padded)
    sentiment_score = np.argmax(output_sentimientos)

    # Sentiment mapping
    sentiment_labels = {0: "joy", 1: "sadness", 2: "support", 3: "funny", 4: "thanks", 5: "goodbye"}
    sentiment = sentiment_labels.get(sentiment_score, "neutral")

    # Generate response
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
