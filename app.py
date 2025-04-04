from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)
score_resp = []  # Declaring globally

@app.get("/")
def index_get():
    return render_template("index.html", request=request)

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    print("Received Message:", text)  # Debugging Step

    if not text:
        return jsonify({"answer": "Error: No input received!"})

    response = get_response(text)  # Pass only one argument
    print("Bot Response:", response)  # Debugging Step

    if not response:
        return jsonify({"answer": "Error: No response from model!"})

    score_resp.append(response[1][0])  # Score tracking
    response = response[0]

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
