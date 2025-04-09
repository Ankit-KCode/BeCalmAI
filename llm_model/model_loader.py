import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🚀 model_loader.py is executing...")

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("📂 sys.path updated:", sys.path)

# Load model configuration
try:
    from llm_model.config import MODEL_NAME, USE_CUDA
    print(f"✅ Config loaded: MODEL_NAME = {MODEL_NAME}, USE_CUDA = {USE_CUDA}")
except Exception as e:
    print(f"❌ Failed to load config: {e}")
    sys.exit(1)

def load_local_model():
    print(f"🔄 Loading local model: {MODEL_NAME} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # Move to GPU if available and enabled
        if USE_CUDA and torch.cuda.is_available():
            model = model.to("cuda")
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ GPU not available or disabled — using CPU")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model '{MODEL_NAME}': {e}")
        sys.exit(1)

# Test model loading directly if this file is run standalone
if __name__ == "__main__":
    model, tokenizer = load_local_model()
    print("✅ Local model test load successful")
