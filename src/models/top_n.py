"""Top-N most-played songs (global or per-user)."""

from __future__ import annotations

import pandas as pd

from src.data.loader import MySpotifyRecommender


def top_n_songs(
    rs: MySpotifyRecommender, n: int = 10, user_id: str | None = None
) -> pd.DataFrame:
    """Top *n* songs by total play count (or by a single user's own plays).

    ``(user_id, song_id)`` is unique in the triplets table, so the per-user
    path needs no groupby.
    """
    if user_id is None:
        top = (
            rs.triplets[["song_id", "play_count"]]
            .groupby("song_id", as_index=False)["play_count"]
            .sum()
            .nlargest(n, "play_count")
        )
    else:
        top = rs.triplets[rs.triplets["user_id"] == user_id][["song_id", "play_count"]].nlargest(
            n, "play_count"
        )

    df = top.merge(
        rs.tracks.drop_duplicates("song_id")[["song_id", "track_id", "artist", "title"]],
        on="song_id",
        how="left",
    )
    return df[["track_id", "artist", "title", "play_count"]]
