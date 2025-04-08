from flask import Flask, request, jsonify, render_template
import torch
from llm_model.model_loader import load_llama_model

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model & tokenizer once at startup
print("🔄 Loading LLaMA model...")
model, tokenizer = load_llama_model()
print("✅ LLaMA model loaded!")
print("👋 llama_chat.py is running")
def generate_response(user_input):
    # Encode input and generate response
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )

    # Decode generated text
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"answer": "Please type something so I can respond."})

        response = generate_response(user_input)
        return jsonify({"answer": response})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"answer": "Oops! Something went wrong on the server."})


# ✅ Keep only ONE entry point
if __name__ == "__main__":
    print("🚀 Starting Flask server at http://127.0.0.1:5050 ...")
    app.run(debug=True, host="127.0.0.1", port=5050)

