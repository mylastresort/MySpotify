"""Raw-file parsers for the Million Song Dataset (MSD)."""

from pathlib import Path

import pandas as pd


def parse_tracks(path: Path) -> pd.DataFrame:
    """Parse ``p02_unique_tracks.txt`` (sep='<SEP>')."""
    return pd.read_csv(
        path,
        sep="<SEP>",
        header=None,
        names=["track_id", "song_id", "artist", "title"],
        engine="python",
    )


def parse_genres(path: Path) -> pd.DataFrame:
    """Parse ``p02_msd_tagtraum_cd2.cls`` (tab-separated, comment='#')."""
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["track_id", "genre_major", "genre_minor"],
    )


def parse_triplets(path: Path) -> pd.DataFrame:
    """Parse ``train_triplets.txt`` (tab-separated)."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "song_id", "play_count"],
    )
    df["play_count"] = df["play_count"].astype(int)
    return df


def parse_lyrics(path: Path) -> pd.DataFrame:
    """Parse MXM sparse bag-of-words into long/tidy format (track_id, word, count)."""
    vocab_raw: list[str] = []
    data_lines: list[str] = []
    in_vocab = False

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.startswith("%") or in_vocab:
                chunk = line.lstrip("%")
                vocab_raw.append(chunk)
                in_vocab = True
                if ":" in line and not line.startswith("%"):
                    data_lines.append(line)
                    in_vocab = False
                    vocab_raw.pop()
            else:
                if in_vocab:
                    in_vocab = False
                data_lines.append(line)

    vocabulary = ",".join(vocab_raw).split(",")
    if vocabulary and vocabulary[0] in ("%i", "i", ""):
        vocabulary = vocabulary[1:]

    rows: list[dict] = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split(",")
        track_id = parts[0]
        for pair in parts[1:]:
            if ":" not in pair:
                continue
            idx_str, cnt_str = pair.split(":", 1)
            idx = int(idx_str) - 1
            if 0 <= idx < len(vocabulary):
                rows.append(
                    {"track_id": track_id, "word": vocabulary[idx], "count": int(cnt_str)}
                )
    return pd.DataFrame(rows, columns=["track_id", "word", "count"])
