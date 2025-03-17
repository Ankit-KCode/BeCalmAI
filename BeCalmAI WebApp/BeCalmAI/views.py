# from django.shortcuts import render

# Create your views here. ------------------------------------//
import openai
from django.shortcuts import render
from django.http import JsonResponse
import json

# Replace with your OpenAI API key
openai.api_key = "your_openai_api_key"

def chat_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "")

        # OpenAI API call to generate AI response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )

        ai_response = response["choices"][0]["message"]["content"]

        return JsonResponse({"response": ai_response})

    return render(request, "BeCalmAI/chat.html")
# -------------------------------------------------------//