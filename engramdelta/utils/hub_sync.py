"""Optional persistence through a private Hugging Face Hub dataset repo.

Enabled only when the environment has HF_REPO (e.g. "user/engramdelta-runs") and HF_TOKEN. In the notebooks
the token comes from Kaggle Secrets. Without them every function is a no-op, and results persist only through
Kaggle notebook outputs. Sync errors are printed, never raised: a flaky upload must not kill a training run.
"""
import os
from typing import Optional


def enabled() -> bool:
    return bool(os.environ.get("HF_REPO")) and bool(os.environ.get("HF_TOKEN"))


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def ensure_repo():
    if not enabled():
        return
    try:
        _api().create_repo(os.environ["HF_REPO"], repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        print(f"[hub] create_repo failed: {e}")


def push_file(local_path: str, path_in_repo: str) -> bool:
    if not enabled():
        return False
    try:
        _api().upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                           repo_id=os.environ["HF_REPO"], repo_type="dataset")
        return True
    except Exception as e:
        print(f"[hub] upload of {local_path} failed: {e}")
        return False


def push_folder(local_dir: str, path_in_repo: str) -> bool:
    if not enabled() or not os.path.isdir(local_dir):
        return False
    try:
        _api().upload_folder(folder_path=local_dir, path_in_repo=path_in_repo,
                             repo_id=os.environ["HF_REPO"], repo_type="dataset")
        return True
    except Exception as e:
        print(f"[hub] folder upload of {local_dir} failed: {e}")
        return False


def download_file(path_in_repo: str, local_path: str) -> bool:
    if not enabled():
        return False
    try:
        from huggingface_hub import hf_hub_download
        tmp_dir = os.path.join(os.path.dirname(local_path) or ".", ".hub_tmp")
        p = hf_hub_download(os.environ["HF_REPO"], path_in_repo, repo_type="dataset",
                            token=os.environ["HF_TOKEN"], local_dir=tmp_dir)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        os.replace(p, local_path)
        return True
    except Exception:
        return False


def pull_prefix(prefix: str, local_dir: str) -> Optional[str]:
    """Download every repo file under `prefix` into local_dir (keeping repo paths)."""
    if not enabled():
        return None
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(os.environ["HF_REPO"], repo_type="dataset", token=os.environ["HF_TOKEN"],
                                 allow_patterns=[f"{prefix.rstrip('/')}/**"], local_dir=local_dir)
    except Exception as e:
        print(f"[hub] pull of {prefix} failed: {e}")
        return None
