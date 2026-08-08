"""Data loader — discovers raw files, builds CSVs, loads into DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.parsers import parse_genres, parse_lyrics, parse_tracks, parse_triplets


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

    # ── CSV builder ──────────────────────────────────────────────────────

    @classmethod
    def _build_csvs(cls, out_dir: Path, root: Path) -> None:
        """Parse raw MSD files and write the 4 CSVs to *out_dir*."""
        out_dir.mkdir(parents=True, exist_ok=True)
        tracks_raw = cls._find_raw_file(root, "p02_unique_tracks.txt")
        genres_raw = cls._find_raw_file(root, "p02_msd_tagtraum_cd2.cls")
        triplets_raw = cls._find_raw_file(root, "train_triplets.txt")
        lyrics_raw = cls._find_raw_file(root, "mxm_dataset_train.txt")

        missing_raw = [
            n
            for n, p in [
                ("tracks", tracks_raw),
                ("genres", genres_raw),
                ("triplets", triplets_raw),
                ("lyrics", lyrics_raw),
            ]
            if p is None
        ]
        if missing_raw:
            raise FileNotFoundError(f"Raw source files not found: {missing_raw}")

        steps = [
            ("tracks.csv", "Tracks", lambda: parse_tracks(tracks_raw)),
            ("genres.csv", "Genres", lambda: parse_genres(genres_raw)),
            ("triplets.csv", "Triplets", lambda: parse_triplets(triplets_raw)),
            ("lyrics.csv", "Lyrics", lambda: parse_lyrics(lyrics_raw)),
        ]
        for filename, label, parser in steps:
            dest = out_dir / filename
            print(f"  [{label}] parsing...", end=" ", flush=True)
            df = parser()
            df.to_csv(dest, index=False)
            print(f"{df.shape}  ->  {dest.name}  ({dest.stat().st_size / 1e6:.0f} MB)")

    # ── constructor ──────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls, triplets_sample_rows: int | None = None
    ) -> MySpotifyRecommender:
        """Load the 4 DataFrames from CSVs (build from raw if needed).

        Parameters
        ----------
        triplets_sample_rows : int, optional
            If given, only read this many rows from triplets.csv (useful for
            quick iteration during development).
        """
        is_kaggle = Path("/kaggle/input").exists()
        cwd = Path.cwd()
        root = cwd.parent if (cwd.name == "notebooks" and not is_kaggle) else cwd

        _needed = {"tracks.csv", "genres.csv", "triplets.csv", "lyrics.csv"}

        def _find_csv_dir() -> Path | None:
            if is_kaggle:
                hits = list(Path("/kaggle/input").rglob("tracks.csv"))
                if hits:
                    candidate = hits[0].parent
                    if all((candidate / f).exists() for f in _needed):
                        return candidate
                return None
            return root / "data" / "csv"

        csv_dir = _find_csv_dir()
        if csv_dir is None or not all((csv_dir / f).exists() for f in _needed):
            if is_kaggle:
                raise FileNotFoundError(
                    "CSVs not found under /kaggle/input. "
                    "Add 'mylastresort/p02-myspotify' as a dataset source."
                )
            csv_dir = root / "data" / "csv"
            missing = [f for f in _needed if not (csv_dir / f).exists()]
            if missing:
                print(f"Missing CSVs {missing} -- building from raw files ...")
                cls._build_csvs(csv_dir, root)

        print(f"Environment : {'Kaggle' if is_kaggle else 'Local'}")
        print(f"CSV dir     : {csv_dir}")

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

        print(f"\n  tracks      {tracks.shape}")
        print(f"  genres      {genres.shape}")
        print(f"  triplets    {triplets.shape}")
        print(f"  lyrics_long {lyrics_long.shape}")

        return cls(
            tracks=tracks,
            genres=genres,
            triplets=triplets,
            lyrics_long=lyrics_long,
        )

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
