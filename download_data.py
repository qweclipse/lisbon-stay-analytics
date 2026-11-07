"""Download the audited snapshot; never replace a file before SHA-256 validation."""
from hashlib import file_digest
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def verified(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        return False
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest() == expected["sha256"]


def main():
    sources = json.loads((ROOT / "output/data_audit.json").read_text(encoding="utf-8"))["files"]
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("listings", "reviews", "calendar"):
        expected = sources[name]
        target = raw / f"{name}.csv.gz"
        if verified(target, expected):
            print(f"{name}: existing SHA-256 verified")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        with urlopen(expected["url"], timeout=120) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if not verified(temporary, expected):
            raise ValueError(f"{name}: checksum mismatch; original file left unchanged")
        temporary.replace(target)
        print(f"{name}: downloaded and SHA-256 verified")


if __name__ == "__main__":
    main()
