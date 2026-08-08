"""01 — Top N Most-Played Songs."""

from __future__ import annotations

import pandas as pd

from src.data.loader import MySpotifyRecommender


def top_n_songs(rs: MySpotifyRecommender, n: int = 250) -> pd.DataFrame:
    """Return the *n* globally most-played songs ranked by total play count.

    Algorithm
    ---------
    1. Aggregate ``play_count`` (sum) per ``song_id``.
    2. Rank descending, keep top *n*.
    3. Enrich with artist / title from ``tracks``.

    Returns
    -------
    DataFrame with columns: rank (index), artist, title, play_count
    """
    song_plays = rs.song_plays()
    enriched = rs.enrich(song_plays)

    result = enriched.head(n).reset_index(drop=True)
    result.index += 1
    result.index.name = "rank"
    return result[["artist", "title", "play_count"]]
