"""04 — Collaborative Filtering (user-based + item-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.data.loader import MySpotifyRecommender
from src.utils.metrics import precision_at_k


def train_test_split(
    rs: MySpotifyRecommender,
    test_ratio: float = 0.2,
    min_test: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split triplets: last ``test_ratio`` of each user's interactions -> test."""
    train_parts, test_parts = [], []
    for _, grp in rs.triplets.groupby("user_id"):
        grp_sorted = grp.sort_values("play_count", ascending=False)
        n_test = max(1, int(len(grp_sorted) * test_ratio))
        if n_test < min_test and len(grp_sorted) >= min_test:
            n_test = min_test
        if len(grp_sorted) <= 1:
            train_parts.append(grp_sorted)
        else:
            train_parts.append(grp_sorted.iloc[n_test:])
            test_parts.append(grp_sorted.iloc[:n_test])
    train = pd.concat(train_parts, ignore_index=True)
    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=rs.triplets.columns)
    )
    print(f"Train: {train.shape[0]:,} rows  |  Test: {test.shape[0]:,} rows")
    return train, test


def build_user_item_matrix(
    train: pd.DataFrame,
) -> tuple[sparse.csr_matrix, dict, dict, dict]:
    """Build a sparse user x song matrix from training data.

    Returns
    -------
    user_item : csr_matrix
    user_idx  : dict[str, int]
    song_idx  : dict[str, int]
    idx_song  : dict[int, str]
    """
    users = train["user_id"].unique()
    songs = train["song_id"].unique()
    user_idx = {u: i for i, u in enumerate(users)}
    song_idx = {s: i for i, s in enumerate(songs)}
    idx_song = {i: s for s, i in song_idx.items()}

    rows = train["user_id"].map(user_idx).values
    cols = train["song_id"].map(song_idx).values
    vals = train["play_count"].values.astype(np.float32)

    user_item = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(users), len(songs))
    )
    print(f"Training user-item matrix: {user_item.shape}  (nnz={user_item.nnz:,})")
    return user_item, user_idx, song_idx, idx_song


# ── User-based CF ────────────────────────────────────────────────────────


def recommend_users(
    user_id: str,
    user_item: sparse.csr_matrix,
    user_idx: dict,
    idx_song: dict,
    k_neighbors: int = 50,
    top_n: int = 10,
) -> list[str]:
    """User-based CF: return top_n song_ids for the given user."""
    uid = user_idx[user_id]
    user_vec = user_item[uid]

    sims = cosine_similarity(user_vec, user_item).flatten()
    sims[uid] = -1

    top_neighbors = np.argsort(sims)[-k_neighbors:]
    neighbor_sims = sims[top_neighbors]

    weighted = neighbor_sims[:, None] * user_item[top_neighbors].toarray()
    scores = weighted.sum(axis=0)

    heard = user_item[uid].nonzero()[1]
    scores[heard] = -1

    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [idx_song[i] for i in top_indices]


def recommend_users_df(
    user_id: str,
    user_item: sparse.csr_matrix,
    user_idx: dict,
    idx_song: dict,
    tracks_df: pd.DataFrame,
    k_neighbors: int = 50,
    top_n: int = 10,
) -> pd.DataFrame:
    """User-based CF: return a display DataFrame with artist, title, score."""
    uid = user_idx[user_id]
    user_vec = user_item[uid]

    sims = cosine_similarity(user_vec, user_item).flatten()
    sims[uid] = -1

    top_neighbors = np.argsort(sims)[-k_neighbors:]
    neighbor_sims = sims[top_neighbors]

    weighted = neighbor_sims[:, None] * user_item[top_neighbors].toarray()
    scores = weighted.sum(axis=0)

    heard = user_item[uid].nonzero()[1]
    scores[heard] = -1

    top_indices = np.argsort(scores)[-top_n:][::-1]
    top_songs = [idx_song[i] for i in top_indices]
    top_scores = scores[top_indices]

    tracks_dedup = tracks_df.drop_duplicates("song_id")[["song_id", "artist", "title"]]
    rec_df = pd.DataFrame({"song_id": top_songs, "score": top_scores})
    rec_df = rec_df.merge(tracks_dedup, on="song_id", how="left")
    rec_df.index += 1
    rec_df.index.name = "rank"
    return rec_df[["artist", "title", "score"]]


# ── Item-based CF ────────────────────────────────────────────────────────


def recommend_tracks(
    song_id: str,
    user_item: sparse.csr_matrix,
    song_idx: dict,
    idx_song: dict,
    top_n: int = 10,
) -> list[str]:
    """Item-based CF: return top_n similar song_ids for the given song."""
    sid = song_idx[song_id]
    song_vec = user_item[:, sid].T

    sims = cosine_similarity(song_vec, user_item.T).flatten()
    sims[sid] = -1

    top_indices = np.argsort(sims)[-top_n:][::-1]
    return [idx_song[i] for i in top_indices]


def recommend_tracks_df(
    song_id: str,
    user_item: sparse.csr_matrix,
    song_idx: dict,
    idx_song: dict,
    tracks_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Item-based CF: return a display DataFrame with artist, title, score."""
    sid = song_idx[song_id]
    song_vec = user_item[:, sid].T

    sims = cosine_similarity(song_vec, user_item.T).flatten()
    sims[sid] = -1

    top_indices = np.argsort(sims)[-top_n:][::-1]
    top_songs = [idx_song[i] for i in top_indices]
    top_sims = sims[top_indices]

    tracks_dedup = tracks_df.drop_duplicates("song_id")[["song_id", "artist", "title"]]
    rec_df = pd.DataFrame({"song_id": top_songs, "score": top_sims})
    rec_df = rec_df.merge(tracks_dedup, on="song_id", how="left")
    rec_df.index += 1
    rec_df.index.name = "rank"
    return rec_df[["artist", "title", "score"]]


# ── Evaluation ───────────────────────────────────────────────────────────


def evaluate_user_cf(
    rs: MySpotifyRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_item: sparse.csr_matrix,
    user_idx: dict,
    idx_song: dict,
    max_users: int = 2000,
) -> float:
    """Evaluate user-based CF with Precision@10."""
    test_by_user = test.groupby("user_id")["song_id"].apply(set).to_dict()
    eval_users = [u for u in test_by_user if u in user_idx and len(test_by_user[u]) > 0]
    np.random.seed(42)
    if len(eval_users) > max_users:
        eval_users = list(np.random.choice(eval_users, max_users, replace=False))

    print(f"Evaluating {len(eval_users)} users...")
    precisions = []
    for i, uid in enumerate(eval_users):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(eval_users)}")
        try:
            rec_songs = recommend_users(uid, user_item, user_idx, idx_song, top_n=10)
            relevant = test_by_user[uid]
            pk = precision_at_k(rec_songs, relevant, k=10)
            precisions.append(pk)
        except Exception:
            pass

    avg_pk = np.mean(precisions) if precisions else 0
    print(f"\nAverage Precision@10 (user-based): {avg_pk:.4f} ({avg_pk*100:.1f}%)")
    print(f"Target: > 10%  ->  {'PASS' if avg_pk > 0.10 else 'FAIL'}")
    return avg_pk


def evaluate_item_cf(
    rs: MySpotifyRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_item: sparse.csr_matrix,
    song_idx: dict,
    idx_song: dict,
    max_users: int = 1000,
) -> float:
    """Evaluate item-based CF with Precision@10."""
    test_by_user_song = test.groupby("user_id")["song_id"].apply(set).to_dict()
    train_by_user_song = train.groupby("user_id")["song_id"].apply(set).to_dict()

    eval_users = [
        u
        for u in test_by_user_song
        if u in {uid for uid in train["user_id"].unique()}
        and len(test_by_user_song[u]) > 0
    ]
    np.random.seed(42)
    if len(eval_users) > max_users:
        eval_users = list(np.random.choice(eval_users, max_users, replace=False))

    print(f"Evaluating {len(eval_users)} users (item-based)...")
    precisions_item = []
    for i, uid in enumerate(eval_users):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(eval_users)}")
        try:
            train_songs = list(train_by_user_song.get(uid, set()))
            test_songs = test_by_user_song[uid]

            candidate_counts: dict[str, int] = {}
            for sid in train_songs[:20]:
                if sid not in song_idx:
                    continue
                rec_songs = recommend_tracks(
                    sid, user_item, song_idx, idx_song, top_n=10
                )
                for s in rec_songs:
                    candidate_counts[s] = candidate_counts.get(s, 0) + 1

            ranked = sorted(candidate_counts, key=candidate_counts.get, reverse=True)
            recommended = ranked[:10]

            relevant = test_songs - set(train_songs)
            pk = precision_at_k(recommended, relevant, k=10)
            precisions_item.append(pk)
        except Exception:
            pass

    avg_pk_item = np.mean(precisions_item) if precisions_item else 0
    print(
        f"\nAverage Precision@10 (item-based): {avg_pk_item:.4f} ({avg_pk_item*100:.1f}%)"
    )
    print(f"Target: > 10%  ->  {'PASS' if avg_pk_item > 0.10 else 'FAIL'}")
    return avg_pk_item
