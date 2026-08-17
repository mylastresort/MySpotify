"""Raw-file parsers and loader for the Million Song Dataset (MSD)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_tracks(path: Path) -> pl.DataFrame:
    """Parse ``p02_unique_tracks.txt`` (sep='<SEP>'; polars can't split on multi-char separators)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(line.rstrip("\n").split("<SEP>"))
    return pl.DataFrame(rows, schema=["track_id", "song_id", "artist", "title"], orient="row")


def parse_genres(path: Path) -> pl.DataFrame:
    """Parse ``p02_msd_tagtraum_cd2.cls`` (tab-separated, '#' comments)."""
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        new_columns=["track_id", "genre_major", "genre_minor"],
    )


def parse_triplets(path: Path) -> pl.DataFrame:
    """Parse ``train_triplets.txt`` (tab-separated).

    user_id/song_id are cast to Categorical at parse time to keep the
    in-memory footprint around 600 MB instead of ~7 GB.
    """
    return (
        pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=["user_id", "song_id", "play_count"],
        )
        .with_columns(
            pl.col("user_id").cast(pl.Categorical),
            pl.col("song_id").cast(pl.Categorical),
        )
    )


def parse_lyrics(path: Path, chunk_size: int = 500_000) -> pl.DataFrame:
    """Parse MXM sparse bag-of-words into long/tidy format (track_id, word, count).

    Reads in chunks to avoid holding the full ~16 M row list in memory.
    """
    vocab: list[str] = []
    chunk: list[tuple] = []
    chunks: list[pl.DataFrame] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("%"):
                vocab = line.lstrip("%").split(",")
                continue
            track_id, *pairs = line.split(",")
            for pair in pairs:
                if ":" not in pair:
                    continue
                idx, cnt = pair.split(":", 1)
                i = int(idx) - 1
                if 0 <= i < len(vocab):
                    chunk.append((track_id, vocab[i], int(cnt)))
                    if len(chunk) >= chunk_size:
                        chunks.append(pl.DataFrame(chunk, schema=["track_id", "word", "count"], orient="row"))
                        chunk = []
    if chunk:
        chunks.append(pl.DataFrame(chunk, schema=["track_id", "word", "count"], orient="row"))
    return pl.concat(chunks)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

RAW_FILENAMES = {
    "tracks": "p02_unique_tracks.txt",
    "genres": "p02_msd_tagtraum_cd2.cls",
    "triplets": "train_triplets.txt",
    "lyrics": "mxm_dataset_train.txt",
}


@dataclass
class MySpotifyRecommender:
    tracks: pl.DataFrame
    genres: pl.DataFrame
    triplets: pl.DataFrame
    lyrics_long: pl.DataFrame

    @classmethod
    def from_files(
        cls,
        data_dir: str | Path,
        triplets_sample_rows: int | None = None,
        download: bool = True,
    ) -> MySpotifyRecommender:
        raw = Path(data_dir) / "raw"
        missing = [name for name in RAW_FILENAMES.values() if not (raw / name).exists()]
        if missing:
            if not download:
                raise FileNotFoundError(
                    "No raw data found. Run './download_dataset.sh uncleaned' to fetch it."
                )
            root = Path(data_dir).parent
            subprocess.run(
                ["bash", "download_dataset.sh", "uncleaned"], check=True, cwd=root
            )

        genres = parse_genres(raw / RAW_FILENAMES["genres"]).rename(
            {"genre_major": "majority_genre", "genre_minor": "minority_genre"}
        )
        triplets = parse_triplets(raw / RAW_FILENAMES["triplets"])
        if triplets_sample_rows is not None:
            triplets = triplets.head(triplets_sample_rows)
        rs = cls(
            tracks=parse_tracks(raw / RAW_FILENAMES["tracks"]),
            genres=genres,
            triplets=triplets,
            lyrics_long=parse_lyrics(raw / RAW_FILENAMES["lyrics"]),
        )
        # Ensure song_id is Categorical everywhere so joins with triplets work.
        rs.tracks = rs.tracks.with_columns(pl.col("song_id").cast(pl.Categorical))
        for name, df in [
            ("tracks", rs.tracks),
            ("genres", rs.genres),
            ("triplets", rs.triplets),
            ("lyrics_long", rs.lyrics_long),
        ]:
            print(f"  {name:<11} {df.shape}")
        return rs
