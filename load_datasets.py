import os
import shutil
import subprocess
from pathlib import Path

# ── path config ──────────────────────────────────────────────────────────────
DATASET_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "mylastresort/p02-myspotify")
NOTEBOOK_DIR = Path.cwd()


def _candidate_roots():
    roots = []
    env_dir = os.getenv("MYSPOTIFY_DATA_DIR") or os.getenv("DATA_DIR")
    if env_dir:
        roots.append(Path(env_dir).expanduser())

    roots.extend(
        [
            NOTEBOOK_DIR,
            NOTEBOOK_DIR / "data",
            NOTEBOOK_DIR / "dataset",
            NOTEBOOK_DIR / "datasets",
            NOTEBOOK_DIR.parent,
            Path("/kaggle/input"),
            Path("/kaggle/working"),
        ]
    )

    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _resolve_file(*relative_paths):
    for root in _candidate_roots():
        if not root.exists():
            continue

        for relative_path in relative_paths:
            candidate = root / relative_path
            if candidate.exists():
                return candidate

        for relative_path in relative_paths:
            matches = list(root.glob(f"**/{relative_path}"))
            if matches:
                return matches[0]

    return None


def _download_from_kaggle_if_needed(target_root):
    if shutil.which("kaggle") is None:
        return None

    target_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset '{DATASET_SLUG}' to {target_root} ...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(target_root),
            "--unzip",
        ],
        check=True,
    )
    return target_root


def load_datasets():
    TRACKS_FILE = _resolve_file("p02_unique_tracks.txt")
    GENRES_FILE = _resolve_file("p02_msd_tagtraum_cd2.cls")
    TRIPLETS_FILE = _resolve_file(
        "p02_train_triplets.txt/train_triplets.txt", "train_triplets.txt"
    )
    MXM_FILE = _resolve_file(
        "p02_mxm_dataset_train.txt/mxm_dataset_train.txt", "mxm_dataset_train.txt"
    )

    if not all([TRACKS_FILE, GENRES_FILE, TRIPLETS_FILE, MXM_FILE]):
        download_root = Path(
            os.getenv("MYSPOTIFY_DATA_DIR", NOTEBOOK_DIR / "data" / "P02. MySpotify")
        ).expanduser()
        try:
            _download_from_kaggle_if_needed(download_root)
        except Exception as exc:
            print(f"Kaggle download skipped or failed: {exc}")

        TRACKS_FILE = TRACKS_FILE or _resolve_file("p02_unique_tracks.txt")
        GENRES_FILE = GENRES_FILE or _resolve_file("p02_msd_tagtraum_cd2.cls")
        TRIPLETS_FILE = TRIPLETS_FILE or _resolve_file(
            "p02_train_triplets.txt/train_triplets.txt", "train_triplets.txt"
        )
        MXM_FILE = MXM_FILE or _resolve_file(
            "p02_mxm_dataset_train.txt/mxm_dataset_train.txt", "mxm_dataset_train.txt"
        )

    missing = [
        name
        for name, path in {
            "TRACKS_FILE": TRACKS_FILE,
            "GENRES_FILE": GENRES_FILE,
            "TRIPLETS_FILE": TRIPLETS_FILE,
            "MXM_FILE": MXM_FILE,
        }.items()
        if path is None
    ]

    if missing:
        raise FileNotFoundError(
            "Could not resolve the dataset files. Set MYSPOTIFY_DATA_DIR to the dataset folder or provide Kaggle credentials for the download fallback. "
            f'Missing: {", ".join(missing)}'
        )

    DATA_DIR = TRACKS_FILE.parent

    print("Data directory:", DATA_DIR)
    for f in [TRACKS_FILE, GENRES_FILE, TRIPLETS_FILE, MXM_FILE]:
        size = f.stat().st_size / 1e6 if f.exists() else -1
        status = f"{size:.1f} MB" if size >= 0 else "❌ NOT FOUND"
        print(f"  {f.name:45s} {status}")

    return TRACKS_FILE, GENRES_FILE, TRIPLETS_FILE, MXM_FILE
