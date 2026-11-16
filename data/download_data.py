"""
Dataset and Backbone Weights Downloader
Compatible across Windows, Linux, and macOS.
"""
import os
import sys
import hashlib
import urllib.request
import zipfile
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PRETRAINED_DIR = os.path.join(ROOT_DIR, "pretrained")

ZENODO_URL = "https://zenodo.org/records/5579392/files/PolypDataset.zip?download=1"
EXPECTED_ZIP_MD5 = "0c45c453bc16424df7a2528966ccd939"

BACKBONE_URL = "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth"
EXPECTED_SHA256_PREFIX = "3cf99910"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, target_path):
    print(f"Downloading {url} -> {target_path}...")
    urllib.request.urlretrieve(url, target_path)


def setup_backbone():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    target = os.path.join(PRETRAINED_DIR, "res2net50_v1b_26w_4s-3cf99910.pth")
    if os.path.exists(target) and sha256(target).startswith(EXPECTED_SHA256_PREFIX):
        print("Backbone weights verified and ready.")
        return
    download_file(BACKBONE_URL, target)
    assert sha256(target).startswith(EXPECTED_SHA256_PREFIX), "Backbone SHA256 mismatch!"
    print("Backbone download verified successfully.")


def setup_dataset():
    train_img = os.path.join(DATA_DIR, "TrainDataset", "image")
    test_kvasir = os.path.join(DATA_DIR, "TestDataset", "Kvasir")
    if os.path.isdir(train_img) and os.path.isdir(test_kvasir):
        print("Dataset already present in data/. Skipping download.")
        return

    zip_path = os.path.join(DATA_DIR, "PolypDataset.zip")
    if not os.path.exists(zip_path):
        download_file(ZENODO_URL, zip_path)

    print("Verifying MD5 checksum of PolypDataset.zip...")
    actual_md5 = md5(zip_path)
    assert actual_md5 == EXPECTED_ZIP_MD5, f"MD5 mismatch: got {actual_md5}, expected {EXPECTED_ZIP_MD5}"

    print("Extracting PolypDataset.zip...")
    extract_dir = os.path.join(DATA_DIR, "polyp_raw")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    os.makedirs(os.path.join(DATA_DIR, "TrainDataset"), exist_ok=True)
    # Move files
    raw_test = os.path.join(extract_dir, "TestDataset", "TestDataset")
    if not os.path.exists(raw_test):
        raw_test = os.path.join(extract_dir, "TestDataset")
    if os.path.exists(raw_test):
        shutil.move(raw_test, os.path.join(DATA_DIR, "TestDataset"))

    shutil.move(os.path.join(extract_dir, "NewTRimage"), os.path.join(DATA_DIR, "TrainDataset", "image"))
    shutil.move(os.path.join(extract_dir, "NewTRmask"), os.path.join(DATA_DIR, "TrainDataset", "masks"))

    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("Dataset setup completed successfully.")


if __name__ == "__main__":
    setup_backbone()
    setup_dataset()
