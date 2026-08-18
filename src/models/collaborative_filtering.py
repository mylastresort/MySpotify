"""Collaborative filtering (implicit ALS) on the user x song matrix."""

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from scipy import sparse

from src.data import MySpotifyRecommender


def build_user_item_matrix(rs: MySpotifyRecommender):
    triplets = rs.triplets
    users = triplets["user_id"].cat.get_categories().to_list()
    songs = triplets["song_id"].cat.get_categories().to_list()
    u_codes = triplets["user_id"].cast(pl.UInt32).to_numpy().astype(np.int64)
    s_codes = triplets["song_id"].cast(pl.UInt32).to_numpy().astype(np.int64)
    user_idx = {u: i for i, u in enumerate(users)}
    song_idx = {s: i for i, s in enumerate(songs)}
    idx_song = {i: s for s, i in song_idx.items()}
    mat = sparse.csr_matrix(
        (triplets["play_count"].to_numpy().astype(np.float64), (u_codes, s_codes)),
        shape=(len(users), len(songs)),
    )
    return mat, user_idx, song_idx, idx_song


def fit_als(
    user_item: sparse.csr_matrix,
    factors: int = 50,
    regularization: float = 0.1,
    alpha: float = 40.0,
    iterations: int = 20,
    use_gpu: bool = False,
    random_state: int = 42,
    num_threads: int = 8,
) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        use_gpu=use_gpu,
        random_state=random_state,
        num_threads=num_threads,
    )
    model.fit(user_item, show_progress=True)
    return model


def evaluate_user_cf(
    model: AlternatingLeastSquares,
    user_item: sparse.csr_matrix,
    user_item_test: sparse.csr_matrix,
    K: int = 10,
) -> float:
    return precision_at_k(model, user_item, user_item_test, K=K, show_progress=True)

def recommend_users_df(
    user_id: str,
    model: AlternatingLeastSquares,
    user_item: sparse.csr_matrix,
    user_idx: dict[str, int],
    idx_song: dict[int, str],
    tracks_df: pl.DataFrame,
    top_n: int = 10,
) -> pl.DataFrame:
    uid = user_idx[user_id]
    item_ids, scores = model.recommend(uid, user_item[uid], N=top_n, filter_already_liked_items=True)
    tracks = tracks_df.unique(subset="song_id", keep="first").select(
        "song_id", "artist", "title"
    )
    recs = pl.DataFrame({"song_id": [idx_song[i] for i in item_ids], "score": scores})
    if tracks["song_id"].dtype == pl.Categorical:
        recs = recs.with_columns(pl.col("song_id").cast(pl.Categorical))
    return (
        recs.join(tracks, on="song_id", how="left")
        .select("artist", "title")
        .rename({"artist": "artist_name", "title": "track_title"})
        .with_row_index("index_number", offset=0)
    )


def recommend_tracks_df(
    song_id: str,
    model: AlternatingLeastSquares,
    song_idx: dict[str, int],
    idx_song: dict[int, str],
    tracks_df: pl.DataFrame,
    top_n: int = 10,
) -> pl.DataFrame:
    sid = song_idx[song_id]
    item_ids, scores = model.similar_items(sid, N=top_n + 1)
    mask = item_ids != sid
    item_ids, scores = item_ids[mask][:top_n], scores[mask][:top_n]
    tracks = tracks_df.unique(subset="song_id", keep="first").select(
        "song_id", "artist", "title"
    )
    recs = pl.DataFrame({"song_id": [idx_song[i] for i in item_ids], "score": scores})
    recs = recs.with_columns(pl.col("song_id").cast(tracks["song_id"].dtype))
    return (
        recs.join(tracks, on="song_id", how="left")
        .with_row_index("rank", offset=0)
        .select("rank", "artist", "title")
    )
