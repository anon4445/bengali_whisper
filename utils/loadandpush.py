from datasets import load_dataset
from huggingface_hub import Repository

repo_name = "bn"  
username = "emon-j"  
repo_url = f"https://huggingface.co/datasets/{username}/{repo_name}"

bn_dataset = load_dataset("bn")
print(bn_dataset)
local_repo_path = "./local_bn_repo" 
ja_dataset.save_to_disk(local_repo_path)
repo = Repository(local_repo_path, clone_from=repo_url)
repo.git_add(auto_lfs_track=True)  
repo.git_commit("Initial commit: Adding dataset")
repo.git_push()
print(f"Dataset successfully pushed to {repo_url}")
