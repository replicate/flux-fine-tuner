from collections import deque
import hashlib
import os
import shutil
import subprocess
import time


class WeightsDownloadCache:
    def __init__(
        self, min_disk_free: int = 10 * (2**30), base_dir: str = "/src/weights-cache"
    ):
        """
        WeightsDownloadCache is meant to track and download weights files as fast
        as possible, while ensuring there's enough disk space.

        It tries to keep the most recently used weights files in the cache, so
        ensure you call ensure() on the weights each time you use them.

        It will not re-download weights files that are already in the cache.

        :param min_disk_free: Minimum disk space required to start download, in bytes.
        :param base_dir: The base directory to store weights files.
        """
        self.min_disk_free = min_disk_free
        self.base_dir = base_dir
        self._hits = 0
        self._misses = 0

        # Least Recently Used (LRU) cache for paths
        self.lru_paths = deque()
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def _remove_least_recent(self) -> None:
        """
        Remove the least recently used weights file from the cache and disk.
        """
        oldest = self.lru_paths.popleft()
        self._rm_disk(oldest)

    def cache_info(self) -> str:
        """
        Get cache information.

        :return: Cache information.
        """

        return f"CacheInfo(hits={self._hits}, misses={self._misses}, base_dir='{self.base_dir}', currsize={len(self.lru_paths)})"

    def _rm_disk(self, path: str) -> None:
        """
        Remove a weights file or directory from disk.
        :param path: Path to remove.
        """
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def _has_enough_space(self) -> bool:
        """
        Check if there's enough disk space.

        :return: True if there's more than min_disk_free free, False otherwise.
        """
        disk_usage = shutil.disk_usage(self.base_dir)
        print(f"Free disk space: {disk_usage.free}")
        return disk_usage.free >= self.min_disk_free

    def ensure(self, url: str) -> str:
        """
        Ensure weights file is in the cache and return its path.

        This also updates the LRU cache to mark the weights as recently used.

        :param url: URL to download weights file from, if not in cache.
        :return: Path to weights.
        """
        path = self.weights_path(url)

        if path in self.lru_paths:
            # here we remove to re-add to the end of the LRU (marking it as recently used)
            self._hits += 1
            self.lru_paths.remove(path)
        else:
            self._misses += 1
            self.download_weights(url, path)

        self.lru_paths.append(path)  # Add file to end of cache
        return path

    def weights_path(self, url: str) -> str:
        """
        Generate path to store a weights file based hash of the URL.

        :param url: URL to download weights file from.
        :return: Path to store weights file.
        """
        hashed_url = hashlib.sha256(url.encode()).hexdigest()
        short_hash = hashed_url[:16]  # Use the first 16 characters of the hash
        return os.path.join(self.base_dir, short_hash)

    def download_weights(self, url: str, dest: str) -> None:
        """
        Download weights file from a URL, ensuring there's enough disk space.

        :param url: URL to download weights file from.
        :param dest: Path to store weights file.
        """
        print("Ensuring enough disk space...")
        while not self._has_enough_space() and len(self.lru_paths) > 0:
            self._remove_least_recent()

        print(f"Downloading weights: {url}")

        st = time.time()
        # maybe retry with the real url if this doesn't work
        try:
            output = subprocess.check_output(["pget", "-x", url, dest], close_fds=True)
            print(output)
        except subprocess.CalledProcessError as e:
            # If download fails, clean up and re-raise exception
            print(e.output)
            self._rm_disk(dest)
            raise e
        print(f"Downloaded weights in {time.time() - st} seconds")
