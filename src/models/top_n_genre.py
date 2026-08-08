"""02 — Top N Most-Played Songs for a Given Genre."""

from __future__ import annotations

import pandas as pd

from src.data.loader import MySpotifyRecommender


def top_n_per_genre(
    rs: MySpotifyRecommender, genre: str, n: int = 100
) -> pd.DataFrame:
    """Return the *n* most-played songs for a given majority genre.

    Algorithm
    ---------
    1. Aggregate ``play_count`` (sum) per ``song_id``.
    2. Join with ``tracks`` to get track_id, artist, title.
    3. Join with ``genres`` via track_id, filter to *genre*.
    4. Keep top *n* by total play count.

    Returns
    -------
    DataFrame with columns: rank (index), artist, title, play_count
    """
    song_plays = rs.song_plays()
    enriched = rs.enrich(song_plays)

    genres_dedup = rs.genres.drop_duplicates("track_id")[
        ["track_id", "majority_genre"]
    ]
    merged = enriched.merge(genres_dedup, on="track_id", how="left")

    filtered = merged[merged["majority_genre"] == genre]
    result = (
        filtered.sort_values("play_count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result.index += 1
    result.index.name = "rank"
    return result[["artist", "title", "play_count"]]
