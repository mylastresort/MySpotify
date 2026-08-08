"""Tests for src.models.top_n and src.models.top_n_genre."""

import numpy as np
import pandas as pd

from src.data.loader import MySpotifyRecommender
from src.models.top_n import top_n_songs
from src.models.top_n_genre import top_n_per_genre


def _make_dummy_rs() -> MySpotifyRecommender:
    """Build a tiny MySpotifyRecommender with synthetic data."""
    tracks = pd.DataFrame(
        {
            "track_id": ["t1", "t2", "t3", "t4"],
            "song_id": ["s1", "s2", "s3", "s4"],
            "artist": ["Artist A", "Artist B", "Artist C", "Artist D"],
            "title": ["Song 1", "Song 2", "Song 3", "Song 4"],
        }
    )
    genres = pd.DataFrame(
        {
            "track_id": ["t1", "t2", "t3", "t4"],
            "majority_genre": ["Rock", "Rock", "Pop", "Jazz"],
            "minority_genre": ["Pop", "Pop", "Rock", "Pop"],
        }
    )
    triplets = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "song_id": ["s1", "s2", "s2", "s3", "s1"],
            "play_count": [10, 5, 8, 3, 20],
        }
    )
    lyrics_long = pd.DataFrame(
        {"track_id": [], "word": [], "count": []}
    ).astype({"track_id": str, "word": str, "count": int})

    return MySpotifyRecommender(
        tracks=tracks, genres=genres, triplets=triplets, lyrics_long=lyrics_long
    )


def test_top_n_songs():
    rs = _make_dummy_rs()
    result = top_n_songs(rs, n=2)
    assert len(result) == 2
    assert result.index[0] == 1
    assert "play_count" in result.columns
    assert result.iloc[0]["play_count"] >= result.iloc[1]["play_count"]


def test_top_n_per_genre():
    rs = _make_dummy_rs()
    result = top_n_per_genre(rs, "Rock", n=10)
    assert len(result) == 2  # t1 and t2 are Rock
    assert all(
        rs.tracks[rs.tracks["song_id"] == s].iloc[0]["artist"] in result["artist"].values
        for s in ["s1", "s2"]
    )
