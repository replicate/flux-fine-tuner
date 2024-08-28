import base64
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import deque
from io import BytesIO
from pathlib import Path

import requests

DEFAULT_CACHE_BASE_DIR = Path("/src/weights-cache")


class WeightsDownloadCache:
    def __init__(
        self, min_disk_free: int = 10 * (2**30), base_dir: Path = DEFAULT_CACHE_BASE_DIR
    ):
        self.min_disk_free = min_disk_free
        self.base_dir = base_dir
        self.hits = 0
        self.misses = 0

        # Least Recently Used (LRU) cache for paths
        self.lru_paths = deque()
        base_dir.mkdir(parents=True, exist_ok=True)

    def ensure(self, url: str) -> Path:
        path = self._weights_path(url)

        if path in self.lru_paths:
            # here we remove to re-add to the end of the LRU (marking it as recently used)
            self.hits += 1
            self.lru_paths.remove(path)
        else:
            self.misses += 1

            while not self._has_enough_space() and len(self.lru_paths) > 0:
                self._remove_least_recent()

            download_weights(url, path)

        self.lru_paths.append(path)  # Add file to end of cache
        return path

    def cache_info(self) -> str:
        return f"CacheInfo(hits={self.hits}, misses={self.misses}, base_dir='{self.base_dir}', currsize={len(self.lru_paths)})"

    def _remove_least_recent(self) -> None:
        oldest = self.lru_paths.popleft()
        print("removing oldest", oldest)
        oldest.unlink()

    def _has_enough_space(self) -> bool:
        disk_usage = shutil.disk_usage(self.base_dir)

        free = disk_usage.free
        print(f"{free=}")  # TODO(andreas): remove debug

        return free >= self.min_disk_free

    def _weights_path(self, url: str) -> Path:
        hashed_url = hashlib.sha256(url.encode()).hexdigest()
        short_hash = hashed_url[:16]  # Use the first 16 characters of the hash
        return self.base_dir / short_hash


def download_weights(url: str, path: Path):
    download_url = make_download_url(url)
    download_weights_url(download_url, path)


def download_weights_url(url: str, path: Path):
    path = Path(path)

    print("Downloading weights")
    start_time = time.time()

    if url.startswith("data:"):
        download_data_url(url, path)
    elif url.endswith(".tar"):
        download_safetensors_tarball(url, path)
    elif url.endswith(".safetensors") or "://civitai.com/api/download" in url:
        download_safetensors(url, path)
    elif url.endswith("/_weights"):
        download_safetensors_tarball(url, path)
    else:
        raise ValueError("URL must end with either .tar or .safetensors")

    print(f"Downloaded weights in {time.time() - start_time:.2f}s")


def find_safetensors(directory: Path) -> list[Path]:
    safetensors_paths = []
    for root, _, files in os.walk(directory):
        root = Path(root)
        for filename in files:
            path = root / filename
            if path.suffix == ".safetensors":
                safetensors_paths.append(path)
    return safetensors_paths


def download_safetensors_tarball(url: str, path: Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        extract_dir = temp_dir / "weights"

        try:
            subprocess.run(["pget", "-x", url, extract_dir], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download tarball: {e}")

        safetensors_paths = find_safetensors(extract_dir)
        if not safetensors_paths:
            raise ValueError("No .safetensors file found in tarball")
        if len(safetensors_paths) > 1:
            raise ValueError("Multiple .safetensors files found in tarball")
        safetensors_path = safetensors_paths[0]

        shutil.move(safetensors_path, path)


def download_data_url(url: str, path: Path):
    _, encoded = url.split(",", 1)
    data = base64.b64decode(encoded)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(fileobj=BytesIO(data), mode="r:*") as tar:
            tar.extractall(path=temp_dir)

        safetensors_paths = find_safetensors(Path(temp_dir))
        if not safetensors_paths:
            raise ValueError("No .safetensors file found in data URI")
        if len(safetensors_paths) > 1:
            raise ValueError("Multiple .safetensors files found in data URI")
        safetensors_path = safetensors_paths[0]

        shutil.move(safetensors_path, path)


def download_safetensors(url: str, path: Path):
    try:
        subprocess.run(["pget", url, str(path)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download safetensors file: {e}")


def make_download_url(url: str) -> str:
    if url.startswith("data:"):
        return url
    if m := re.match(r"^(?:https?://)?huggingface\.co/([^/]+)/([^/]+)/?$", url):
        owner, model_name = m.groups()
        return make_huggingface_download_url(owner, model_name)
    if m := re.match(r"^(?:https?://)?civitai\.com/models/(\d+)(?:/[^/?]+)?/?$", url):
        model_id = m.groups()[0]
        return make_civitai_download_url(model_id)
    if m := re.match(r"^((?:https?://)?civitai\.com/api/download/models/.*)$", url):
        return url
    if m := re.match(r"^(https?://.*\.safetensors)(?:\?|$)", url):
        return m.groups()[0]
    if m := re.match(r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/?$", url):
        owner, model_name = m.groups()
        return make_replicate_model_download_url(owner, model_name)
    if m := re.match(
        r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/(?:versions/)?([^/]+)/?$", url
    ):
        owner, model_name, version_id = m.groups()
        return make_replicate_version_download_url(owner, model_name, version_id)
    if m := re.match(r"^(https?://replicate.delivery/.*\.tar)$", url):
        return m.groups()[0]

    if "huggingface.co" in url:
        raise ValueError(
            "Failed to parse HuggingFace URL. Expected huggingface.co/<owner>/<model-name>"
        )
    if "civitai.com" in url:
        raise ValueError(
            "Failed to parse CivitAI URL. Expected civitai.com/models/<id>[/<model-name>]"
        )
    raise ValueError(
        """Failed to parse URL. Expected either:
* Replicate model in the format <owner>/<username> or <owner>/<username>/<version>
* HuggingFace URL in the format huggingface.co/<owner>/<model-name>
* CivitAI URL in the format civitai.com/models/<id>[/<model-name>]
* Arbitrary .safetensors URLs from the Internet"""
    )


def make_replicate_model_download_url(owner: str, model_name: str) -> str:
    return f"https://replicate.com/{owner}/{model_name}/_weights"


def make_replicate_version_download_url(
    owner: str, model_name: str, version_id: str
) -> str:
    return f"https://replicate.com/{owner}/{model_name}/versions/{version_id}/_weights"


def make_huggingface_download_url(owner: str, model_name: str) -> str:
    url = f"https://huggingface.co/api/models/{owner}/{model_name}/tree/main"
    response = requests.get(url)
    response.raise_for_status()

    files = response.json()
    safetensors_files = [f for f in files if f["path"].endswith(".safetensors")]

    if len(safetensors_files) == 0:
        raise ValueError("No .safetensors file found in the repository")
    if len(safetensors_files) > 1:
        raise ValueError("Multiple .safetensors files found in the repository")

    safetensors_path = safetensors_files[0]["path"]
    return (
        f"https://huggingface.co/{owner}/{model_name}/resolve/main/{safetensors_path}"
    )


def make_civitai_download_url(model_id: str) -> str:
    return f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
