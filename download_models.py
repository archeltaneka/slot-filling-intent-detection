import os
import logging

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    REPO_ID = "archeltaneka/slot-filling-intent-detection" 
    LOCAL_CHECKPOINT_DIR = "files/checkpoints_test"
    
    # Models to be downloaded
    MODELS_TO_DOWNLOAD = [
        "jointbilstm_model.pth",
        "jointbilstm_attn_model.pth",
        "bert_model.pth",
        "baseline_model.joblib"
    ]

    # Save directory
    os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)

    for model_file in MODELS_TO_DOWNLOAD:
        try:
            logging.info(f"Downloading {model_file} from {REPO_ID}...")
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=model_file,
                local_dir=LOCAL_CHECKPOINT_DIR,
                local_dir_use_symlinks=False  # Copies the actual file into the folder
            )
            logging.info(f"Successfully downloaded {model_file} to {path}")
        except Exception as e:
            logging.error(f"Failed to download {model_file}: {e}")
