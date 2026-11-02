"""Download the WiSig ManySig subset into data/raw/wisig_manysig/ and verify it."""

import hashlib
import zipfile
from pathlib import Path

import gdown

URL = "https://drive.google.com/uc?id=1szuns8MhcYocdbipK9t9TM9MLgEMklxk"
OUT_DIR = Path(__file__).parent / "raw" / "wisig_manysig"
PKL = OUT_DIR / "ManySig.pkl"
EXPECTED_SHA256 = "2B0A7A7488DD3650BCAE7B1D80EFBCFFD1598AAA671AE6B0A0DF2A24DC0F694F"


def sha256_of(path):
    """Return the SHA-256 code of a file, reading it 1 MB at a time."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path):
    """Stop with an error if the file's SHA-256 code differs from the expected one."""
    actual = sha256_of(path)
    if actual != EXPECTED_SHA256.lower():
        raise SystemExit(f"Wrong file: {path}\n  expected {EXPECTED_SHA256}\n  got      {actual}")
    print(f"OK: {path.name} is the expected file.")


def main():
    # 1. Already there? Only verify it.
    if PKL.exists():
        check(PKL)
        return

    # 2. Download the zip.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "ManySig.zip"
    if gdown.download(URL, str(zip_path), quiet=False) is None:
        raise SystemExit("Download failed. Try again later (Google Drive may be limiting downloads).")

    # 3. Unzip, then delete the zip.
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(OUT_DIR)
    zip_path.unlink()
    if not PKL.exists():
        raise SystemExit(f"Unzipped, but {PKL} is missing. Check the folder structure inside the zip.")

    # 4. Verify.
    check(PKL)


if __name__ == "__main__":
    main()