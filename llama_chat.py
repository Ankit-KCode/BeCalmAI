from flask import Flask, request, jsonify, render_template
from llm_model.model_loader import load_local_model
from llm_model.cloud_model import call_openrouter 
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")

# Loading local model on startup
print("🧠 Initializing local fallback model...")
model, tokenizer = load_local_model()
print("✅ Local model ready!")

# Function to generate reply using local model
def generate_local_reply(user_input):
    try:
        inputs = tokenizer.encode(user_input, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            model.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reply.strip()
    except Exception as e:
        print(f"❌ Local generation error: {e}")
        return "I'm having trouble generating a response right now."

# Routing: Home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


# Routing: Chat prediction
@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"answer": "Please type something so I can respond."})

    try:
        print("🌐 Trying cloud response...")
        response = call_openrouter(user_input)
        if "🌐 Error" in response or not response.strip():
            raise Exception("Cloud failed")
        print("✅ Cloud response successful")
        return jsonify({"answer": response})
    except Exception as e:
        print(f"⚠️ Cloud failed. Using local model. Reason: {e}")
        fallback = generate_local_reply(user_input)
        return jsonify({"answer": fallback})

# Starting the Flask server
if __name__ == "__main__":
    print("🚀 Starting Flask server at http://127.0.0.1:5050 ...")
    app.run(debug=True, host="127.0.0.1", port=5050)
