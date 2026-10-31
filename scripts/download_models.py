"""Download the fastText language-ID model and cache the sentence-transformers model."""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_PATH = MODELS_DIR / "lid.176.bin"
LID_MIN_BYTES = 120_000_000
ST_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def _progress(blocks: int, block_size: int, total: int) -> None:
    done = blocks * block_size
    if total > 0:
        pct = min(100.0, done * 100 / total)
        sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {done / 1e6:7.1f} MB")
    sys.stdout.flush()


def download_fasttext() -> Path:
    """Download lid.176.bin (~126 MB) unless a complete copy exists."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if LID_PATH.exists() and LID_PATH.stat().st_size >= LID_MIN_BYTES:
        print(f"fastText: already present ({LID_PATH.stat().st_size / 1e6:.1f} MB)")
        return LID_PATH
    print(f"fastText: downloading {LID_URL}")
    tmp = LID_PATH.with_suffix(".part")
    urllib.request.urlretrieve(LID_URL, tmp, reporthook=_progress)  # noqa: S310 - fixed https URL
    print()
    if tmp.stat().st_size < LID_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded fastText file is too small; retry.")
    tmp.replace(LID_PATH)
    return LID_PATH


def cache_sentence_transformer() -> Path | None:
    """Load the model once so it is cached locally; return the cache directory if found."""
    from sentence_transformers import SentenceTransformer

    print(f"sentence-transformers: loading {ST_MODEL} (downloads ~1.1 GB on first run)")
    SentenceTransformer(ST_MODEL)
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    matches = list(hub.glob(f"models--*{ST_MODEL}")) if hub.exists() else []
    return matches[0] if matches else None


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    lid = download_fasttext()
    st_dir = cache_sentence_transformer()
    print("\nDone.")
    print(f"  fastText : {lid} ({lid.stat().st_size / 1e6:.1f} MB)")
    if st_dir:
        print(f"  ST model : {st_dir} ({_dir_size(st_dir) / 1e6:.1f} MB)")
    else:
        print("  ST model : cached (location not found under ~/.cache/huggingface/hub)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
