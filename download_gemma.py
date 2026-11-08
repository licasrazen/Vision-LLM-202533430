
from huggingface_hub import hf_hub_download
from pathlib import Path

REPO_ID = "bartowski/google_gemma-4-E2B-it-GGUF"
MODEL_FILE = "google_gemma-4-E2B-it-Q4_K_M.gguf"

MODEL_DIR = Path("src/models/Gemma4")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_FILE,
    local_dir=str(MODEL_DIR)
)

print("다운로드 완료:", model_path)
