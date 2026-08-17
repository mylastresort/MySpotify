"""Top-100 most-played songs for a given genre (majority genre)."""

import polars as pl

from src.data import MySpotifyRecommender


def top_n_per_genre(
    rs: MySpotifyRecommender,
    genre: str,
    n: int = 10,
    user_id: str | None = None,
) -> pl.DataFrame:
    songs = (
        rs.tracks.unique(subset="song_id", keep="first")
        .select("song_id", "track_id")
        .join(
            rs.genres.filter(pl.col("majority_genre") == genre), on="track_id", how="inner"
        )
        .select("song_id")
    )
    if user_id is None:
        top = (
            rs.triplets.join(songs, on="song_id", how="inner")
            .group_by("song_id")
            .agg(pl.col("play_count").sum())
            .sort("play_count", descending=True, maintain_order=True)
            .head(n)
        )
    else:
        top = (
            rs.triplets.filter(pl.col("user_id") == user_id)
            .join(songs, on="song_id", how="inner")
            .select("song_id", "play_count")
            .sort("play_count", descending=True, maintain_order=True)
            .head(n)
        )
    tracks = rs.tracks.unique(subset="song_id", keep="first").select(
        "song_id", "track_id", "artist", "title"
    )
    return top.join(tracks, on="song_id", how="left").select(
        "track_id", "artist", "title", "play_count"
    )
