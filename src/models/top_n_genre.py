"""Top-100 most-played songs for a given genre (majority genre)."""

from __future__ import annotations

import pandas as pd

from src.data.loader import MySpotifyRecommender


def top_n_per_genre(
    rs: MySpotifyRecommender,
    genre: str,
    n: int = 10,
    user_id: str | None = None,
) -> pd.DataFrame:
    """Top *n* most-played songs for *genre* (or a single user's, if user_id given).

    Tracks are deduplicated by ``song_id`` (keeping the first row) *before* the
    genre join, matching the reference output: a song whose genre tag sits on a
    non-canonical track_id is dropped.
    """
    tracks = rs.tracks.drop_duplicates("song_id")
    df = tracks.merge(
        rs.genres[rs.genres["majority_genre"] == genre],
        on="track_id",
    ).merge(
        rs.triplets,
        on="song_id",
        how="left",
    )

    if user_id is None:
        top = (
            df[["song_id", "play_count"]]
            .groupby("song_id", as_index=False)["play_count"]
            .sum()
            .nlargest(n, "play_count")
        )
    else:
        top = df[df["user_id"] == user_id][["song_id", "play_count"]].nlargest(n, "play_count")

    df = top.merge(
        tracks[["song_id", "track_id", "artist", "title"]],
        on="song_id",
        how="left",
    )
    return df[["track_id", "artist", "title", "play_count"]]
