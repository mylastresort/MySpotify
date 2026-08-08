"""Bonus — Three Spotify-Inspired Features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.data.loader import MySpotifyRecommender


def because_you_listened_to(
    rs: MySpotifyRecommender,
    user_id: str,
    top_n: int = 10,
) -> tuple[pd.DataFrame, str]:
    """'Because you listened to [Artist]' -- artist-seeded recommendations.

    1. Find the user's most-played artist.
    2. Collect fans of that artist (other users who played >=1 of their tracks).
    3. Aggregate what those fans play most.
    4. Remove tracks the seed user already knows.
    5. Return top_n by fan play score.
    """
    if user_id not in rs.triplets["user_id"].values:
        raise KeyError(f"Unknown user_id: {user_id!r}")

    user_plays = (
        rs.triplets[rs.triplets["user_id"] == user_id]
        .merge(rs.tracks[["song_id", "artist"]], on="song_id", how="left")
    )
    top_artist = user_plays.groupby("artist")["play_count"].sum().idxmax()

    artist_songs = rs.tracks[rs.tracks["artist"] == top_artist]["song_id"].unique()

    fans = rs.triplets[rs.triplets["song_id"].isin(artist_songs)][
        "user_id"
    ].unique()
    fans = fans[fans != user_id]

    fan_plays = (
        rs.triplets[rs.triplets["user_id"].isin(fans)]
        .groupby("song_id", as_index=False)["play_count"]
        .sum()
        .rename(columns={"play_count": "fan_score"})
    )

    known = rs.triplets[rs.triplets["user_id"] == user_id]["song_id"].unique()

    out = (
        fan_plays[~fan_plays["song_id"].isin(known)]
        .merge(rs.tracks[["song_id", "artist", "title"]], on="song_id", how="left")
        .sort_values("fan_score", ascending=False)
        .head(top_n)[["artist", "title", "fan_score"]]
        .reset_index(drop=True)
    )
    out.index += 1
    out.index.name = "rank"
    return out, top_artist


def your_genre_mix(
    rs: MySpotifyRecommender,
    user_id: str,
    top_n: int = 10,
    popularity_weight: float = 0.3,
) -> tuple[pd.DataFrame, str]:
    """'Your [Genre] Mix' -- genre-anchored personalised playlist.

    1. Find the user's dominant genre via play history.
    2. Collect all tracks in that genre.
    3. Score = blend of global popularity + user affinity.
    4. Remove already-heard tracks; return top_n.
    """
    if user_id not in rs.triplets["user_id"].values:
        raise KeyError(f"Unknown user_id: {user_id!r}")

    user_plays = rs.triplets[rs.triplets["user_id"] == user_id]

    user_with_genre = (
        user_plays
        .merge(rs.tracks[["song_id", "track_id"]], on="song_id", how="left")
        .merge(rs.genres[["track_id", "majority_genre"]], on="track_id", how="left")
    )
    genre_totals = (
        user_with_genre.groupby("majority_genre")["play_count"].sum().dropna()
    )
    if genre_totals.empty:
        raise ValueError("Cannot determine dominant genre for this user.")
    top_genre = genre_totals.idxmax()

    genre_track_ids = rs.genres[
        rs.genres["majority_genre"] == top_genre
    ]["track_id"].unique()

    genre_songs = rs.tracks[rs.tracks["track_id"].isin(genre_track_ids)][
        ["song_id", "track_id", "artist", "title"]
    ]

    global_plays = (
        rs.triplets[rs.triplets["song_id"].isin(genre_songs["song_id"])]
        .groupby("song_id", as_index=False)["play_count"]
        .sum()
        .rename(columns={"play_count": "global_plays"})
    )

    user_affinity = (
        user_plays[user_plays["song_id"].isin(genre_songs["song_id"])]
        [["song_id", "play_count"]]
        .rename(columns={"play_count": "user_plays"})
    )

    known = user_plays["song_id"].unique()

    candidates = (
        genre_songs
        .merge(global_plays, on="song_id", how="left")
        .merge(user_affinity, on="song_id", how="left")
        .fillna({"global_plays": 0, "user_plays": 0})
    )

    max_global = candidates["global_plays"].max() or 1
    max_user = candidates["user_plays"].max() or 1
    candidates["score"] = (
        popularity_weight * (candidates["global_plays"] / max_global)
        + (1 - popularity_weight) * (candidates["user_plays"] / max_user)
    )

    out = (
        candidates[~candidates["song_id"].isin(known)]
        .sort_values("score", ascending=False)
        .head(top_n)[["artist", "title", "score"]]
        .reset_index(drop=True)
    )
    out.index += 1
    out.index.name = "rank"
    return out, top_genre


def build_artist_user_matrix(
    rs: MySpotifyRecommender,
) -> tuple[sp.csr_matrix, dict, dict]:
    """Build sparse artist x user matrix from triplets.

    Returns
    -------
    matrix     : csr_matrix
    artist_idx : dict[str, int]
    idx_artist : dict[int, str]
    """
    song_artist = rs.tracks[["song_id", "artist"]].drop_duplicates("song_id")
    artist_plays = (
        rs.triplets.merge(song_artist, on="song_id", how="left")
        .dropna(subset=["artist"])
        .groupby(["artist", "user_id"], as_index=False)["play_count"]
        .sum()
    )

    artist_idx = {a: i for i, a in enumerate(artist_plays["artist"].unique())}
    user_idx = {u: i for i, u in enumerate(artist_plays["user_id"].unique())}

    rows = artist_plays["artist"].map(artist_idx)
    cols = artist_plays["user_id"].map(user_idx)
    data = artist_plays["play_count"].astype(float)

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(artist_idx), len(user_idx)),
    )
    idx_artist = {i: a for a, i in artist_idx.items()}
    return matrix, artist_idx, idx_artist


def fans_also_like(
    rs: MySpotifyRecommender,
    artist_name: str,
    artist_matrix: sp.csr_matrix,
    artist_idx: dict,
    idx_artist: dict,
    top_n: int = 5,
) -> pd.DataFrame:
    """'Fans also like' -- similar artists by overlapping fanbase (cosine similarity)."""
    if artist_name not in artist_idx:
        raise KeyError(f"Artist not found: {artist_name!r}")

    idx = artist_idx[artist_name]
    artist_vec = artist_matrix[idx]

    sims = cosine_similarity(artist_vec, artist_matrix).flatten()
    sims[idx] = -1

    top_indices = sims.argsort()[::-1][:top_n]
    similar_artists = [idx_artist[i] for i in top_indices]
    similarity_scores = [sims[i] for i in top_indices]

    song_plays = rs.song_plays()
    song_artist = rs.tracks[["song_id", "artist", "title"]].drop_duplicates("song_id")
    top_tracks = (
        song_plays.merge(song_artist, on="song_id", how="left").dropna(subset=["artist"])
    )

    rows = []
    for artist, score in zip(similar_artists, similarity_scores):
        best = top_tracks[top_tracks["artist"] == artist].head(1)
        title = best["title"].iloc[0] if not best.empty else "---"
        rows.append(
            {"similar_artist": artist, "top_track": title, "similarity": round(score, 4)}
        )

    out = pd.DataFrame(rows)
    out.index += 1
    out.index.name = "rank"
    return out
