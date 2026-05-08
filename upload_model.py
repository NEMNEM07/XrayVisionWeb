from huggingface_hub import HfApi

api = HfApi()

# 레포 생성
api.create_repo(repo_id="NEMNEM0702/xrayvision", repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj=r"C:\Projects\XrayVision\checkpoints\best_model.pt",
    path_in_repo="best_model.pt",
    repo_id="NEMNEM0702/xrayvision",
    repo_type="model"
)