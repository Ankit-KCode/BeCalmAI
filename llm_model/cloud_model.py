import requests

# Example using OpenRouter.ai
OPENROUTER_API_KEY = "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"

def call_openrouter(user_input):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": user_input}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "🌐 Error: Could not get smart reply from cloud model."
