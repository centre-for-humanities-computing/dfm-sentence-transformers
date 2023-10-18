import logging
import os
import shutil
import stat
import tempfile

from huggingface_hub import HfApi, HfFolder, Repository
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def save_to_hub(
    model: SentenceTransformer,
    repo_name: str,
    commit_message: str = "Add new SentenceTransformer model.",
    exist_ok: bool = False,
    replace_model_card: bool = False,
):
    """Had to write this function because of a bug in sentence_transformers
    that has not been fixed in any release.
    """
    token = HfFolder.get_token()
    if token is None:
        raise ValueError(
            "You must login to the Hugging Face hub on this computer by typing `transformers-cli login`."
        )

    repo_url = HfApi().create_repo(
        repo_id=repo_name,
        token=token,
        repo_type=None,
        exist_ok=exist_ok,
    )
    _, org, name = repo_url.rsplit("/", 2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # First create the repo (and clone its content if it's nonempty).
        repo = Repository(tmp_dir, clone_from=repo_url)

        create_model_card = replace_model_card or not os.path.exists(
            os.path.join(tmp_dir, "README.md")
        )
        model.save(
            tmp_dir,
            model_name=f"{org}/{name}",
            create_model_card=create_model_card,
        )

        # Find files larger 5M and track with git-lfs
        large_files = []
        for root, dirs, files in os.walk(tmp_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, tmp_dir)

                if os.path.getsize(file_path) > (5 * 1024 * 1024):
                    large_files.append(rel_path)

        if len(large_files) > 0:
            logger.info(
                "Track files with git lfs: {}".format(", ".join(large_files))
            )
            repo.lfs_track(large_files)

        logger.info("Push model to the hub. This might take a while")
        push_return = repo.push_to_hub(commit_message=commit_message)

        def on_rm_error(func, path, exc_info):
            # path contains the path of the file that couldn't be removed
            # let's just assume that it's read-only and unlink it.
            try:
                os.chmod(path, stat.S_IWRITE)
                os.unlink(path)
            except:
                pass

        # Remove .git folder. On Windows, the .git folder might be read-only and cannot be deleted
        # Hence, try to set write permissions on error
        try:
            for f in os.listdir(tmp_dir):
                shutil.rmtree(os.path.join(tmp_dir, f), onerror=on_rm_error)
        except Exception as e:
            logger.warning(
                "Error when deleting temp folder: {}".format(str(e))
            )
            pass

    return push_return
