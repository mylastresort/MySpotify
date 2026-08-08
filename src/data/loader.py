"""Data loader — resolve cleaned/uncleaned data paths and load into DataFrames.

Path model
----------
``from_files`` takes a *data_dir* holding the dataset:
``data_dir/raw``  : uncleaned raw MSD text files (downloaded via ``download_dataset.sh uncleaned``)
``data_dir/csv``  : cleaned CSVs (downloaded via ``download_dataset.sh cleaned``,
                    or written by ``from_files`` when it parses the raw files)

``from_files`` checks ``data_dir/csv`` first and loads the parsed CSVs; only when
they are missing does it parse the uncleaned raw files, save the parsed CSVs
to ``data_dir/csv``, and return them.
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
CSV_FILENAMES = ["tracks.csv", "genres.csv", "triplets.csv", "lyrics.csv"]


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

    # ── data path resolution ─────────────────────────────────────────────

    @staticmethod
    def _raw_dir(data_dir: Path) -> Path:
        return data_dir / "raw"

    @staticmethod
    def _csv_dir(data_dir: Path) -> Path:
        return data_dir / "csv"

    @classmethod
    def _find_csv_dir(cls, root: Path) -> Path | None:
        """Locate the cleaned CSVs directory (``root/csv``, ``root``, or Kaggle input)."""
        if Path("/kaggle/input").exists():
            for hit in Path("/kaggle/input").rglob("tracks.csv"):
                if all((hit.parent / f).exists() for f in CSV_FILENAMES):
                    return hit.parent
            return None
        for candidate in (cls._csv_dir(root), root):
            if all((candidate / f).exists() for f in CSV_FILENAMES):
                return candidate
        return None

    @classmethod
    def _write_csvs(cls, out_dir: Path, rs: MySpotifyRecommender) -> None:
        """Persist the parsed DataFrames as cleaned CSVs (parsed output)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        rs.tracks.to_csv(out_dir / "tracks.csv", index=False)
        rs.genres.to_csv(out_dir / "genres.csv", index=False)
        rs.triplets.to_csv(out_dir / "triplets.csv", index=False)
        rs.lyrics_long.to_csv(out_dir / "lyrics.csv", index=False)

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
    def from_raw(cls, root: Path, triplets_sample_rows: int | None = None) -> MySpotifyRecommender:
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
    def from_csvs(cls, csv_dir: Path, triplets_sample_rows: int | None = None) -> MySpotifyRecommender:
        """Load the cleaned CSVs directly from *csv_dir* (no parsing)."""
        missing = [f for f in CSV_FILENAMES if not (csv_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Cleaned CSVs not found: {missing}")
        print(f"Source      : cleaned CSVs ({csv_dir})")

        tracks = pd.read_csv(csv_dir / "tracks.csv")
        genres = pd.read_csv(csv_dir / "genres.csv").rename(
            columns={"genre_major": "majority_genre", "genre_minor": "minority_genre"}
        )
        triplets = pd.read_csv(
            csv_dir / "triplets.csv",
            dtype={"play_count": "int32"},
            nrows=triplets_sample_rows,
        )
        lyrics_long = pd.read_csv(csv_dir / "lyrics.csv")
        return cls(tracks, genres, triplets, lyrics_long)

    @classmethod
    def from_files(
        cls,
        data_dir: str | Path,
        triplets_sample_rows: int | None = None,
        download: bool = True,
    ) -> MySpotifyRecommender:
        """Load data from *data_dir* (holds ``csv/`` and ``raw/`` subfolders).

        CSV-first: if ``data_dir/csv`` holds all four parsed files, load them
        directly. Only when they are missing are the uncleaned raw files in
        ``data_dir/raw`` parsed and the parsed CSVs written to ``data_dir/csv``
        for the next run. When the raw files are also missing and ``download``
        is True, the dataset is fetched with ``download_dataset.sh`` first.
        """
        data_dir = Path(data_dir)
        print(f"Data dir    : {data_dir}")

        csv_dir = cls._find_csv_dir(data_dir)
        if csv_dir is not None:
            rs = cls.from_csvs(csv_dir, triplets_sample_rows)
        else:
            csv_dir = cls._csv_dir(data_dir)
            raw_dir = cls._raw_dir(data_dir)
            raw_present = all(
                cls._find_raw_file(raw_dir, name) is not None for name in RAW_FILENAMES.values()
            )
            if not raw_present:
                if not download:
                    raise FileNotFoundError(
                        "No data found. Run './download_dataset.sh uncleaned' for the "
                        "raw files or './download_dataset.sh cleaned' for the CSVs."
                    )
                cls._download_raw(data_dir)
                raw_present = all(
                    cls._find_raw_file(raw_dir, name) is not None
                    for name in RAW_FILENAMES.values()
                )
                if not raw_present:
                    raise FileNotFoundError(
                        f"Download finished but raw files still missing under {data_dir}."
                    )
            rs = cls.from_raw(raw_dir)
            if not Path("/kaggle/input").exists():
                cls._write_csvs(csv_dir, rs)
                print(f"Saved parsed CSVs to {csv_dir}")
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
