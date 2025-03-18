# from django.shortcuts import render

# Create your views here. ------------------------------------//
import openai
import nltk
from django.shortcuts import render
from django.http import JsonResponse
# import json


# Load sentiment analysis module
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# OpenAI API Key (Replace with your actual API key)
openai.api_key = "your-openai-api-key"

def chatbot_response(user_input):
    """
    Process user input, analyze sentiment, and generate AI response.
    """
    # Perform sentiment analysis
    sentiment_score = sia.polarity_scores(user_input)
    sentiment = "neutral"
    
    if sentiment_score["compound"] >= 0.05:
        sentiment = "positive"
    elif sentiment_score["compound"] <= -0.05:
        sentiment = "negative"

    # Generate AI response based on sentiment
    response = f"Your sentiment is {sentiment}. Here are some relaxation tips:\n"
    
    if sentiment == "negative":
        response += "Try deep breathing exercises or guided meditation."
    elif sentiment == "positive":
        response += "Keep up the good work! Stay mindful and enjoy the moment."
    else:
        response += "How about a short break with relaxing music?"

    return response

def chat_view(request):
    """
    Renders the chat page.
    """
    return render(request, 'BeCalmAI/chat.html')

def chat_api(request):
    """
    Handles AJAX requests from frontend and returns AI response.
    """
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        bot_response = chatbot_response(user_message)
        return JsonResponse({"response": bot_response})





# Replace with your OpenAI API key -------------------------------
# openai.api_key = "your_openai_api_key"

# def chat_view(request):
#     if request.method == "POST":
#         data = json.loads(request.body)
#         user_message = data.get("message", "")

#         # OpenAI API call to generate AI response
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": user_message}]
#         )

#         ai_response = response["choices"][0]["message"]["content"]

#         return JsonResponse({"response": ai_response})

#     return render(request, "BeCalmAI/chat.html")
# -------------------------------------------------------//