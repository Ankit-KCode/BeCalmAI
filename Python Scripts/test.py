from flask import Flask

print("👋 mini_test.py loaded")

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Flask!"

if __name__ == "__main__":
    print("🚀 Starting Mini Flask Server...")
    app.run(debug=True, port=5050)
