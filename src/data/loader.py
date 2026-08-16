"""Data loader — resolve the uncleaned raw MSD files and parse them into DataFrames.

Path model
----------
``from_files`` takes a *data_dir* holding the dataset:
``data_dir/raw``  : uncleaned raw MSD text files (downloaded via ``download_dataset.sh uncleaned``)

All four tables (tracks, genres, triplets, lyrics) are parsed **in place** from
the raw files with pandas — nothing is cleaned, and no intermediate files are
ever written or read.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.parsers import parse_genres, parse_lyrics, parse_tracks, parse_triplets

# ── well-known file names ────────────────────────────────────────────────

RAW_FILENAMES = {
    "tracks": "p02_unique_tracks.txt",
    "genres": "p02_msd_tagtraum_cd2.cls",
    "triplets": "train_triplets.txt",
    "lyrics": "mxm_dataset_train.txt",
}


@dataclass
class MySpotifyRecommender:
    """Container for the 4 MSD DataFrames + all recommendation methods."""

    tracks: pd.DataFrame
    genres: pd.DataFrame
    triplets: pd.DataFrame
    lyrics_long: pd.DataFrame

    # ── file discovery ───────────────────────────────────────────────────

    @staticmethod
    def _find_raw_file(root: Path, *names: str) -> Path | None:
        """Search *root* and common sub-dirs for a raw dataset file."""
        candidates = [root, root / "data"]
        if (root / "data").exists():
            candidates += [p for p in (root / "data").iterdir() if p.is_dir()]
        for base in candidates:
            if not base.exists():
                continue
            for name in names:
                p = base / name
                if p.exists():
                    return p
            for name in names:
                hits = list(base.glob(f"**/{name}"))
                if hits:
                    return hits[0]
        return None

    @staticmethod
    def _raw_dir(data_dir: Path) -> Path:
        return data_dir / "raw"

    # ── constructors ─────────────────────────────────────────────────────

    @staticmethod
    def _download_raw(data_dir: Path) -> None:
        """Run ``download_dataset.sh uncleaned`` (from *data_dir*'s parent)."""
        root = data_dir.parent
        script = root / "download_dataset.sh"
        if not script.exists():
            raise FileNotFoundError(
                f"Download script not found at {script}; run "
                "'./download_dataset.sh uncleaned' manually."
            )
        print(f"Raw files missing — downloading via {script} uncleaned ...")
        subprocess.run(["bash", str(script), "uncleaned"], check=True, cwd=root)

    @classmethod
    def from_raw(
        cls, root: Path, triplets_sample_rows: int | None = None
    ) -> MySpotifyRecommender:
        """Parse the uncleaned raw MSD files in place — no intermediate files."""
        raw = {key: cls._find_raw_file(root, name) for key, name in RAW_FILENAMES.items()}
        missing = [key for key, path in raw.items() if path is None]
        if missing:
            raise FileNotFoundError(f"Uncleaned raw files not found: {missing}")
        print("Source      : uncleaned raw files (parsed in place)")

        tracks = parse_tracks(raw["tracks"])
        genres = parse_genres(raw["genres"]).rename(
            columns={"genre_major": "majority_genre", "genre_minor": "minority_genre"}
        )
        triplets = parse_triplets(raw["triplets"])
        if triplets_sample_rows is not None:
            triplets = triplets.head(triplets_sample_rows)
        lyrics_long = parse_lyrics(raw["lyrics"])
        return cls(tracks, genres, triplets, lyrics_long)

    @classmethod
    def from_files(
        cls,
        data_dir: str | Path,
        triplets_sample_rows: int | None = None,
        download: bool = True,
    ) -> MySpotifyRecommender:
        """Load data from *data_dir* (holding a ``raw/`` subfolder).

        The uncleaned raw files under ``data_dir/raw`` are parsed in place with
        pandas — nothing is saved to disk. When the raw files are missing and
        ``download`` is True, the dataset is fetched with
        ``download_dataset.sh`` first.
        """
        data_dir = Path(data_dir)
        print(f"Data dir    : {data_dir}")

        raw_dir = cls._raw_dir(data_dir)
        raw_present = all(
            cls._find_raw_file(raw_dir, name) is not None for name in RAW_FILENAMES.values()
        )
        if not raw_present:
            if not download:
                raise FileNotFoundError(
                    "No raw data found. Run './download_dataset.sh uncleaned' to fetch it."
                )
            cls._download_raw(data_dir)
            raw_present = all(
                cls._find_raw_file(raw_dir, name) is not None
                for name in RAW_FILENAMES.values()
            )
            if not raw_present:
                raise FileNotFoundError(
                    f"Download finished but raw files still missing under {raw_dir}."
                )
        rs = cls.from_raw(raw_dir)
        if triplets_sample_rows is not None:
            rs.triplets = rs.triplets.head(triplets_sample_rows)

        print(f"\n  tracks      {rs.tracks.shape}")
        print(f"  genres      {rs.genres.shape}")
        print(f"  triplets    {rs.triplets.shape}")
        print(f"  lyrics_long {rs.lyrics_long.shape}")
        return rs

    # ── helpers ──────────────────────────────────────────────────────────

    def song_plays(self) -> pd.DataFrame:
        """Aggregate total play count per song."""
        return (
            self.triplets.groupby("song_id", as_index=False)["play_count"]
            .sum()
            .sort_values("play_count", ascending=False)
        )

    def enrich(self, song_plays: pd.DataFrame) -> pd.DataFrame:
        """Join song_id -> track_id, artist, title."""
        tracks_dedup = self.tracks.drop_duplicates("song_id")[
            ["song_id", "track_id", "artist", "title"]
        ]
        return song_plays.merge(tracks_dedup, on="song_id", how="inner")
