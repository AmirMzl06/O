import os
import requests
import subprocess
from pathlib import Path

# ==========================================================
# Settings
# ==========================================================

DOI = "10.5061/dryad.cvdncjt7n"

TARGET_FILE = "Jango_ISO_2015.7z"

ROOT = Path("dataset")
DOWNLOAD_DIR = ROOT / "downloads"
EXTRACT_DIR = ROOT / "Jango_ISO_2015"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Find file through Dryad API
# ==========================================================

print("Searching Dryad API...")

url = f"https://datadryad.org/api/v2/datasets/{DOI}/files?page[size]=100"

r = requests.get(url)
r.raise_for_status()

files = r.json()["_embedded"]["stash:files"]

download_url = None

for f in files:

    if f["path"] == TARGET_FILE:

        download_url = f["_links"]["stash:download"]["href"]
        break

if download_url is None:
    raise RuntimeError(f"{TARGET_FILE} not found!")

print("Found!")

# ==========================================================
# Download
# ==========================================================

save_path = DOWNLOAD_DIR / TARGET_FILE

if not save_path.exists():

    print("Downloading...")
    response = requests.get(download_url, stream=True)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(save_path, "wb") as file:

        for chunk in response.iter_content(chunk_size=1024 * 1024):

            if chunk:

                file.write(chunk)
                downloaded += len(chunk)

                if total:

                    percent = downloaded / total * 100
                    print(
                        f"\r{percent:6.2f}% "
                        f"({downloaded/1024**3:.2f}/{total/1024**3:.2f} GB)",
                        end=""
                    )

    print("\nDownload complete.")

else:

    print("Archive already exists.")

# ==========================================================
# Extract
# ==========================================================

print("Extracting...")

try:

    subprocess.run(

        [
            "7z",
            "x",
            str(save_path),
            f"-o{EXTRACT_DIR}",
            "-y",
        ],

        check=True,

    )

    print("Extraction complete.")

except FileNotFoundError:

    print()
    print("=" * 60)
    print("7z is not installed.")
    print()
    print("Ubuntu:")
    print("sudo apt install p7zip-full")
    print()
    print("Windows:")
    print("Install 7-Zip")
    print("https://www.7-zip.org/")
    print("=" * 60)

print("Done.")
