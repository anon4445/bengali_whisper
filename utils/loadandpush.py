from datasets import load_dataset
from huggingface_hub import Repository
import shutil
import os


dataset_folder = "./local_dataset"  
repo_folder = "./local_bn_repo"   
repo_name = "bn"
username = "emon-j"  
repo_url = f"https://huggingface.co/datasets/{username}/{repo_name}"


bn_dataset = load_dataset("bn")


if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder) 
bn_dataset.save_to_disk(dataset_folder)

if os.path.exists(repo_folder):
    shutil.rmtree(repo_folder)  
repo = Repository(repo_folder, clone_from=repo_url)


for item in os.listdir(dataset_folder):
    s = os.path.join(dataset_folder, item)
    d = os.path.join(repo_folder, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

repo.git_add(auto_lfs_track=True)  
repo.git_commit("Initial commit: Adding dataset")
repo.git_push()
