import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests_mock

sys.path.append(str(Path(__file__).parent.parent))
from weights import WeightsDownloadCache, make_download_url


def test_replicate_model_url():
    assert (
        make_download_url("owner/model") == "https://replicate.com/owner/model/_weights"
    )
    assert (
        make_download_url("https://replicate.com/owner/model")
        == "https://replicate.com/owner/model/_weights"
    )


def test_replicate_version_url():
    assert (
        make_download_url("owner/model/version123")
        == "https://replicate.com/owner/model/versions/version123/_weights"
    )
    assert (
        make_download_url("owner/model/versions/version123")
        == "https://replicate.com/owner/model/versions/version123/_weights"
    )
    assert (
        make_download_url("https://replicate.com/owner/model/versions/version123")
        == "https://replicate.com/owner/model/versions/version123/_weights"
    )


def test_replicate_com_url():
    url = "https://replicate.com/owner/model"
    assert make_download_url(url) == "https://replicate.com/owner/model/_weights"


def test_replicate_com_version_url():
    url = "https://replicate.com/owner/model/versions/123abc"
    assert (
        make_download_url(url)
        == "https://replicate.com/owner/model/versions/123abc/_weights"
    )


def test_huggingface_url():
    with requests_mock.Mocker() as m:
        m.get(
            "https://huggingface.co/api/models/owner/model/tree/main",
            json=[{"path": "model.safetensors", "type": "file"}],
        )
        assert (
            make_download_url("https://huggingface.co/owner/model")
            == "https://huggingface.co/owner/model/resolve/main/model.safetensors"
        )


def test_civitai_url():
    assert (
        make_download_url("https://civitai.com/models/12345")
        == "https://civitai.com/api/download/models/12345?type=Model&format=SafeTensor"
    )
    assert (
        make_download_url("civitai.com/models/12345/model-name")
        == "https://civitai.com/api/download/models/12345?type=Model&format=SafeTensor"
    )


def test_direct_safetensors_url():
    assert (
        make_download_url("https://example.com/model.safetensors")
        == "https://example.com/model.safetensors"
    )
    assert (
        make_download_url("https://example.com/model.safetensors?download=true")
        == "https://example.com/model.safetensors"
    )


def test_replicate_delivery_url():
    url = "https://replicate.delivery/pbxt/ABC123/model.tar"
    assert make_download_url(url) == url


def test_data_url():
    data_url = "data:application/x-tar;base64,SGVsbG8gV29ybGQh"
    assert make_download_url(data_url) == data_url


def test_invalid_huggingface_url():
    with pytest.raises(ValueError, match="Failed to parse HuggingFace URL"):
        make_download_url("https://huggingface.co/invalid/url/format")


def test_invalid_civitai_url():
    with pytest.raises(ValueError, match="Failed to parse CivitAI URL"):
        make_download_url("https://civitai.com/invalid/url/format")


def test_unsupported_url():
    with pytest.raises(ValueError, match="Failed to parse URL"):
        make_download_url("https://unsupported.com/model")


def test_huggingface_no_safetensors():
    with requests_mock.Mocker() as m:
        m.get(
            "https://huggingface.co/api/models/owner/model/tree/main",
            json=[{"path": "model.bin", "type": "file"}],
        )
        with pytest.raises(ValueError, match="No .safetensors file found"):
            make_download_url("https://huggingface.co/owner/model")


def test_huggingface_multiple_safetensors():
    with requests_mock.Mocker() as m:
        m.get(
            "https://huggingface.co/api/models/owner/model/tree/main",
            json=[
                {"path": "model1.safetensors", "type": "file"},
                {"path": "model2.safetensors", "type": "file"},
            ],
        )
        with pytest.raises(ValueError, match="Multiple .safetensors files found"):
            make_download_url("https://huggingface.co/owner/model")


@pytest.fixture
def mock_base_dir(tmp_path):
    return tmp_path / "weights-cache"


@pytest.fixture
def cache(mock_base_dir):
    return WeightsDownloadCache(min_disk_free=1000, base_dir=mock_base_dir)


@patch("weights.download_weights")
@patch("shutil.disk_usage")
@patch("pathlib.Path.unlink")
def test_weights_download_cache(
    mock_unlink, mock_disk_usage, mock_download_weights, cache, mock_base_dir
):
    # Setup
    disk_space = [1500, 1000, 500, 1000]  # Simulate changing disk space
    mock_disk_usage.side_effect = [MagicMock(free=space) for space in disk_space]

    # Test ensure method
    url1 = "https://example.com/weights1.tar"
    url2 = "https://example.com/weights2.tar"

    # First call should download
    path1 = cache.ensure(url1)
    mock_download_weights.assert_called_once_with(url1, path1)
    assert path1.parent == mock_base_dir
    assert cache.hits == 0
    assert cache.misses == 1

    # Second call to same URL should hit cache
    cache.ensure(url1)
    assert cache.hits == 1
    assert cache.misses == 1

    # Call with new URL should download again
    _ = cache.ensure(url2)
    assert mock_download_weights.call_count == 2
    assert cache.hits == 1
    assert cache.misses == 2

    # Test LRU behavior
    url3 = "https://example.com/weights3.tar"
    cache.ensure(url3)
    mock_unlink.assert_called_once_with()  # Check that unlink was called

    # Test cache_info
    info = cache.cache_info()
    assert "hits=1" in info
    assert "misses=3" in info
    assert str(mock_base_dir) in info


def test_weights_download_cache_initialization(mock_base_dir):
    cache = WeightsDownloadCache(base_dir=mock_base_dir)
    assert cache.base_dir == mock_base_dir
    assert mock_base_dir.exists()
