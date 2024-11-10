from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/scratch/bjb3az/interaction/good_data",
    repo_id="kingsleykim/test_video",
    repo_type="dataset",
)