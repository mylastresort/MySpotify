import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from scipy import sparse

from src.data.loader import MySpotifyRecommender


def train_test_split(
    rs: MySpotifyRecommender,
    test_ratio: float = 0.2,
    min_test: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions into random train/test (no timestamps in
    the data, so 'last 20%' is approximated by a random holdout)."""
    train_parts, test_parts = [], []
    for _, grp in rs.triplets.groupby("user_id"):
        grp = grp.sample(frac=1, random_state=random_state)
        n_test = max(min_test, int(len(grp) * test_ratio))
        if len(grp) <= n_test:
            train_parts.append(grp)
            continue
        test_parts.append(grp.head(n_test))
        train_parts.append(grp.iloc[n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    train.sort_values(["play_count"], ascending=False, inplace=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=rs.triplets.columns)
    return train, test


def train_test_split_top(
    rs: MySpotifyRecommender,
    test_ratio: float = 0.2,
    min_test: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions by play intensity instead of time.

    For every user the plays are sorted by ``play_count`` descending and the
    top ``test_ratio`` fraction (at least ``min_test``) go to **test**, the rest
    to **train**. There are no timestamps in the data, so the held-out set is
    the user's *strongest* signals rather than their most recent ones. Users too
    small to leave a non-empty train history behind stay entirely in train.

    Vectorised over the whole triplets frame (sort + cumulative count per user)
    so it scales to tens of millions of rows.
    """
    df = rs.triplets.sort_values(
        ["play_count"], ascending=[False], kind="mergesort"
    )

    # sizes = df.groupby("user_id")["play_count"].transform("size")
    # pos = df.groupby("user_id").cumcount()
    # n_test = np.maximum(min_test, (sizes * test_ratio).astype(np.int64))
    # is_test = (pos < n_test) & (sizes > n_test)

    # test = df[is_test]
    # train = df[~is_test]
    # return train, test

    train_parts, test_parts = [], []
    for _, grp in df.groupby("user_id"):
        # grp = grp.sample(frac=1, random_state=random_state)
        n_test = max(min_test, int(len(grp) * test_ratio))
        if len(grp) <= n_test:
            train_parts.append(grp)
            continue
        # test_parts.append(grp.head(n_test))
        # train_parts.append(grp.iloc[n_test:])
        test_parts.append(grp.tail(n_test))
        train_parts.append(grp.iloc[:n_test])
    train = pd.concat(train_parts, ignore_index=True)
    train.sort_values(["play_count"], ascending=False, inplace=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=rs.triplets.columns)
    return train, test


def build_user_item_matrix(
    train: pd.DataFrame,
    # weighting: str = "raw",
) -> tuple[sparse.csr_matrix, dict[str, int], dict[str, int], dict[int, str]]:
    """Build a sparse user x song matrix from training data.

    ``weighting`` controls the entry values (all of them use the real play data):
      - "raw" (default): raw play counts. These are exactly what the implicit
        ALS solver wants: it turns each cell into a confidence
        ``c_ui = alpha * r_ui`` (Hu, Koren & Volinsky 2008), so volume keeps
        its signal.
      - "log": log1p(play_count). Compresses the skew while keeping intensity
        signal, so confidence reflects taste rather than play volume.
      - "tfidf": log1p(play_count) weighted by inverse document frequency, then
        each user row is L2-normalised. The IDF term downweights globally
        popular songs (heard by everyone, so they add noise to the
        neighbourhood); the row norm makes similarity compare listening *shapes*
        rather than total volume.
    """
    user_cat = train["user_id"].astype("category")
    song_cat = train["song_id"].astype("category")

    user_idx = dict(zip(user_cat.cat.categories, range(len(user_cat.cat.categories))))
    song_idx = dict(zip(song_cat.cat.categories, range(len(song_cat.cat.categories))))
    idx_song = dict(zip(range(len(song_cat.cat.categories)), song_cat.cat.categories))

    rows = user_cat.cat.codes.values.astype(np.int32)
    cols = song_cat.cat.codes.values.astype(np.int32)
    # if weighting == "raw":
    vals = train["play_count"].values.astype(np.float32)
    # elif weighting in ("log", "tfidf"):
    #     vals = np.log1p(train["play_count"].values).astype(np.float32)
    # else:
    #     raise ValueError(f"unknown weighting: {weighting!r} (use 'raw', 'log' or 'tfidf')")

    # if weighting == "tfidf":
    #     df = np.bincount(cols, minlength=len(song_cat.cat.categories))
    #     idf = np.log(len(user_cat.cat.categories) / (df + 1)) + 1.0
    #     vals = vals * idf[cols]

    user_item = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_cat.cat.categories), len(song_cat.cat.categories)),
    )
    # if weighting == "tfidf":
    #     norms = np.sqrt(np.asarray(user_item.multiply(user_item).sum(axis=1)).ravel())
    #     norms[norms == 0] = 1.0
    #     user_item = user_item.multiply(1.0 / norms[:, None]).tocsr()
    return user_item, user_idx, song_idx, idx_song

def evaluate_user_cf(
    model: AlternatingLeastSquares,
    user_item: sparse.csr_matrix,
    user_item_test: sparse.csr_matrix,
    top_n: int = 10,
) -> float:
    """Mean precision@top_n of the latent-factor model over test users.

    For each sampled user the model recommends top_n novel songs (train songs
    are filtered out automatically) and the fraction that appear in the user's
    held-out test set is averaged across users.
    """
    pk = precision_at_k(
        model,
        user_item,
        user_item_test,
        K=top_n,
        num_threads=28,
    )
    return pk

def fit_als(
    user_item: sparse.csr_matrix,
    factors: int = 50,
    regularization: float = 0.1,
    alpha: float = 40.0,
    iterations: int = 20,
    use_gpu: bool = False,
    random_state: int = 42,
) -> AlternatingLeastSquares:
    """Fit an implicit feedback ALS model (``implicit.als``) on user x song data.

    Delegates the factorisation to the ``implicit`` library's
    ``AlternatingLeastSquares``: confidence for a play is ``alpha * play_count``
    and preference is 1 iff the user played the song. Returns the fitted model;
    its ``user_factors`` / ``item_factors`` give each user's latent taste and
    each song's latent profile, and ``recommend`` / ``similar_items`` turn them
    into rankings.
    """
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        use_gpu=use_gpu,
        random_state=random_state,
        num_threads=28,
    )
    model.fit(user_item, show_progress=True)
    return model


def recommend_users(
    user_id: str,
    model: AlternatingLeastSquares,
    user_item: sparse.csr_matrix,
    user_idx: dict[str, int],
    idx_song: dict[int, str],
    top_n: int = 10,
) -> list[str]:
    """Return the top_n song_ids recommended to a user by the latent factors.

    ``implicit`` scores every song as ``user_factors[u] @ item_factors[i]`` and
    filters out the songs the user already heard in train.
    """
    uid = user_idx[user_id]
    item_ids, _ = model.recommend(uid, user_item[uid], N=top_n)
    return [idx_song[i] for i in item_ids]


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


def recommend_tracks(
    song_id: str,
    model: AlternatingLeastSquares,
    song_idx: dict[str, int],
    idx_song: dict[int, str],
    top_n: int = 10,
) -> list[str]:
    """Return the top_n song_ids most similar to *song_id* in latent space.

    ``implicit``'s ``similar_items`` returns the seed itself as its most similar
    item, so it is fetched with N = top_n + 1 and dropped.
    """
    sid = song_idx[song_id]
    item_ids, _ = model.similar_items(sid, N=top_n + 1)
    item_ids = item_ids[item_ids != sid][:top_n]
    return [idx_song[i] for i in item_ids]


def recommend_tracks_df(
    song_id: str,
    model: AlternatingLeastSquares,
    song_idx: dict[str, int],
    idx_song: dict[int, str],
    tracks_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Similar-track recommendations as a display DataFrame (rank/artist/title/score)."""
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

def evaluate_item_cf(
    rs: MySpotifyRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: AlternatingLeastSquares,
    user_item: sparse.csr_matrix,
    song_idx: dict[str, int],
    idx_song: dict[int, str],
    top_n: int = 10,
    max_users: int = 1000,
    max_seeds: int = 50,
    verbose: bool = False,
) -> float:
    """Mean precision@top_n for item-based recommendations.

    Each user's top-``max_seeds`` train songs act as seeds; ``similar_items``
    aggregates their most similar songs in latent space. Novelty is enforced by
    zeroing the user's already-heard songs, and precision@top_n is scored
    against the held-out test set.
    """
    test_songs = test.groupby("user_id")["song_id"].apply(set).to_dict()
    train_songs = (
        train.sort_values("play_count", ascending=False)
        .groupby("user_id")["song_id"]
        .apply(list)
        .to_dict()
    )
    eval_users = [
        u
        for u in test_songs
        if train_songs.get(u) and train_songs[u][0] in song_idx
    ]

    np.random.seed(42)
    if len(eval_users) > max_users:
        eval_users = list(np.random.choice(eval_users, max_users, replace=False))

    results = []
    for u in eval_users:
        seeds = [s for s in train_songs[u][:max_seeds] if s in song_idx]
        if not seeds:
            continue
        seed_idx = np.array([song_idx[s] for s in seeds], dtype=int)

        scores = np.zeros(user_item.shape[1], dtype=np.float32)
        for sid in seed_idx:
            ids, sims = model.similar_items(sid, N=top_n)
            scores[ids] += sims

        heard = np.array([song_idx[s] for s in train_songs[u] if s in song_idx], dtype=int)
        scores[heard] = -1

        top = np.argsort(scores)[-top_n:][::-1]
        recs = [idx_song[i] for i in top]
        pk = precision_at_k(recs, test_songs[u], K=top_n)
        results.append((u, pk))
    if verbose:
        for u, pk in sorted(results, key=lambda x: x[1], reverse=True)[:5]:
            print(f"{u}: precision@{top_n} = {pk:.4f}")
    return float(np.mean([pk for _, pk in results])) if results else 0.0
