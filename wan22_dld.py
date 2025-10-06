from huggingface_hub import snapshot_download
import os

# Target directory for local model/tokenizer
target_dir = "models/text_encoders/umt5_xxl_fp16"

# Download the entire repo snapshot to the target directory
snapshot_download(
    repo_id="google/umt5-xxl",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"All files for google/umt5-xxl are now in {target_dir}")