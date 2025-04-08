import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Debug Print: Check if script is running
print("🚀 model_loader.py is executing...")

# Get the absolute path of the root directory (BeCalmAI)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Debug Print: Check if path is set correctly
print("📂 sys.path updated:", sys.path)

# Import config
try:
    from llm_model.config import MODEL_NAME, USE_CUDA
    print(f"✅ Imported config: MODEL_NAME={MODEL_NAME}, USE_CUDA={USE_CUDA}")
except Exception as e:
    print(f"❌ Error importing config: {e}")
    sys.exit(1)

# Function to load LLaMA model
def load_llama_model():
    print("🔄 Loading LLaMA model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        if USE_CUDA and torch.cuda.is_available():
            model = model.to("cuda")
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ Using CPU (slower performance)")

        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

# Call function for debugging
if __name__ == "__main__":
    load_llama_model()
