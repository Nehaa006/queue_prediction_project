import os
import zipfile
import urllib.request

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
DATA_DIR = os.path.join("data", "MOT17")
ZIP_PATH = os.path.join("data", "MOT17.zip")

os.makedirs("data", exist_ok=True)

if not os.path.exists(ZIP_PATH) and not os.path.exists(DATA_DIR):
    print("📥 Downloading official MOT17 dataset (~5.5 GB)... Please wait.")
    urllib.request.urlretrieve(MOT17_URL, ZIP_PATH)
    print("✅ Download completed!")

if os.path.exists(ZIP_PATH) and not os.path.exists(DATA_DIR):
    print("📦 Extracting MOT17 dataset archive...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("✅ Extraction complete! Files placed in data/MOT17/")