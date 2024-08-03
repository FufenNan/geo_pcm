from huggingface_hub import HfApi, hf_hub_download

model_id = "runwayml/stable-diffusion-v1-5"
local_dir = "/home/haoyum3/PCM/Phased-Consistency-Model/sd1.5"  # 指定下载路径

api = HfApi()
files = api.list_repo_files(model_id, repo_type="model", revision="main")
print("文件列表:")
for file in files:
    print(file)

for file in files:
    print(f"Downloading {file}...")
    hf_hub_download(repo_id=model_id, repo_type="model", filename=file, local_dir=local_dir, revision="main")

print("所有文件下载完成。")
