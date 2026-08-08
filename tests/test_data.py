"""Tests for src.data.parsers."""

import tempfile
from pathlib import Path

import pandas as pd

from src.data.parsers import parse_tracks, parse_genres, parse_triplets, parse_lyrics


def test_parse_tracks():
    content = "TRAAAAAB128F42891E1<SEP>SOBNMDV12A8C13F7A0<SEP>Dwight Yoakam<SEP>You're The One\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        df = parse_tracks(Path(f.name))

    assert list(df.columns) == ["track_id", "song_id", "artist", "title"]
    assert len(df) == 1
    assert df.iloc[0]["artist"] == "Dwight Yoakam"


def test_parse_genres():
    content = "TRAAAAAB128F42891E1\tRock\tPop\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cls", delete=False) as f:
        f.write(content)
        f.flush()
        df = parse_genres(Path(f.name))

    assert list(df.columns) == ["track_id", "genre_major", "genre_minor"]
    assert len(df) == 1
    assert df.iloc[0]["genre_major"] == "Rock"


def test_parse_triplets():
    content = "user1\tsong1\t5\nuser2\tsong2\t10\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        df = parse_triplets(Path(f.name))

    assert list(df.columns) == ["user_id", "song_id", "play_count"]
    assert len(df) == 2
    assert df.iloc[0]["play_count"] == 5
    assert df["play_count"].dtype == int
