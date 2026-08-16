"""Collaborative filtering (implicit ALS) on the user x song matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from src.data.loader import MySpotifyRecommender


def build_user_item_matrix(rs: MySpotifyRecommender):
    """User x song sparse matrix plus the id <-> index maps, from the triplets.

    ``(user_id, song_id)`` is unique in the triplets table, so each row of the
    matrix is a (user, song, play_count) triple.
    """
    u_codes, u_uniques = pd.factorize(rs.triplets["user_id"])
    s_codes, s_uniques = pd.factorize(rs.triplets["song_id"])
    user_idx = {u: int(i) for i, u in enumerate(u_uniques)}
    song_idx = {s: int(i) for i, s in enumerate(s_uniques)}
    idx_song = {i: s for s, i in song_idx.items()}
    mat = sparse.csr_matrix(
        (rs.triplets["play_count"].to_numpy().astype(np.float64), (u_codes, s_codes)),
        shape=(len(user_idx), len(song_idx)),
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
    """Fit an implicit-feedback ALS model (Hu, Koren & Volinsky 2008)."""
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
    top_n: int = 10,
    sample_users: int = 50_000,
    random_state: int = 42,
) -> float:
    """Pooled precision@top_n on a random sample of test users.

    Replicates ``implicit.evaluation.precision_at_k``'s exact definition —
    ``sum(hits) / sum(min(top_n, test_items))`` over test users, with
    ``filter_already_liked_items=False`` — so the estimate matches the full
    sweep (implicit's full sweep over ~1 M users takes ~15 min; 50 k users
    give the same value within a couple of 0.1 pp in seconds via a batched
    ``recommend`` call).
    """
    test_users = np.flatnonzero(np.diff(user_item_test.indptr) > 0)
    rng = np.random.default_rng(random_state)
    n = min(sample_users, len(test_users))
    sample = rng.choice(test_users, size=n, replace=False)

    batch_ids, _ = model.recommend(sample, user_item[sample], N=top_n)

    test_indices = user_item_test.indices
    test_indptr = user_item_test.indptr
    counts = np.diff(test_indptr)
    relevant = 0.0
    pr_div = 0.0
    for j, u in enumerate(sample):
        start, stop = test_indptr[u], test_indptr[u + 1]
        test_items = set(test_indices[start:stop].tolist())
        relevant += len(set(batch_ids[j].tolist()) & test_items)
        pr_div += min(top_n, counts[u])
    return relevant / pr_div


def recommend_users_df(
    user_id: str,
    model: AlternatingLeastSquares,
    user_item: sparse.csr_matrix,
    user_idx: dict[str, int],
    idx_song: dict[int, str],
    tracks_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """User recommendations as a display DataFrame (rank/artist/title/score)."""
    uid = user_idx[user_id]
    item_ids, scores = model.recommend(uid, user_item[uid], N=top_n)

    tracks_dedup = tracks_df.drop_duplicates("song_id")[["song_id", "artist", "title"]]
    rec_df = pd.DataFrame({"song_id": [idx_song[i] for i in item_ids], "score": scores})
    rec_df = rec_df.merge(tracks_dedup, on="song_id", how="left")
    rec_df.index += 1
    rec_df.index.name = "rank"
    return rec_df[["artist", "title", "score"]]


def recommend_tracks_df(
    song_id: str,
    model: AlternatingLeastSquares,
    song_idx: dict[str, int],
    idx_song: dict[int, str],
    tracks_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Similar-track recommendations as a display DataFrame (rank/artist/title/score).

    The seed itself is always the most similar item, so it is fetched with
    ``top_n + 1`` neighbours and dropped.
    """
    sid = song_idx[song_id]
    item_ids, scores = model.similar_items(sid, N=top_n + 1)
    mask = item_ids != sid
    item_ids, scores = item_ids[mask][:top_n], scores[mask][:top_n]

    tracks_dedup = tracks_df.drop_duplicates("song_id")[["song_id", "artist", "title"]]
    rec_df = pd.DataFrame({"song_id": [idx_song[i] for i in item_ids], "score": scores})
    rec_df = rec_df.merge(tracks_dedup, on="song_id", how="left")
    rec_df.index += 1
    rec_df.index.name = "rank"
    return rec_df[["artist", "title", "score"]]
