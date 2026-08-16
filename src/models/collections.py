"""Collections: 50 most-played songs about a keyword.

Three approaches over the full in-memory tables (no streaming):
  * baseline        — tracks where the keyword token appears at least ``n`` times
  * word2vec        — keyword expanded with embedding neighbours, then summed
  * classification  — ML classifiers on the TF-IDF sparse lyrics matrix
"""

from __future__ import annotations

import gensim.downloader
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import PorterStemmer
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data.loader import MySpotifyRecommender

_STEMMER = PorterStemmer()


def _vocab_tokens(kw: str, vocab: set[str]) -> list[str]:
    """Map a raw keyword to the tokens present in the (Porter-stemmed) lyrics vocab.

    The musiXmatch vocabulary stores stems, so ``happiness`` / ``loneliness``
    appear as ``happi`` / ``loneli``. An exact vocab match wins; otherwise the
    Porter stem is tried. Keywords matching nothing return ``[]`` (which yields
    an empty result instead of a crash).
    """
    if kw in vocab:
        return [kw]
    stem = _STEMMER.stem(kw)
    return [stem] if stem in vocab else []


def _tracks_plays(rs: MySpotifyRecommender):
    tracks = rs.tracks.drop_duplicates("track_id")[
        ["track_id", "song_id", "artist", "title"]
    ]
    plays = rs.triplets.groupby("song_id", as_index=False)["play_count"].sum()
    return tracks, plays


def _format_result(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return (
        df.sort_values("play_count", ascending=False)[["artist", "title", "play_count"]]
        .head(top_n)
        .pipe(lambda x: x.set_axis(range(1, len(x) + 1)))
    )


def collection_baseline(
    rs: MySpotifyRecommender,
    kw: list[str],
    n: int = 10,
    top_n: int = 50,
) -> dict[str, pd.DataFrame]:
    """Tracks where the keyword token appears at least ``n`` times."""
    vocab = set(rs.lyrics_long["word"].unique())
    tracks, plays = _tracks_plays(rs)

    result: dict[str, pd.DataFrame] = {}
    for k in kw:
        tokens = _vocab_tokens(k, vocab)
        if not tokens:
            result[k] = pd.DataFrame(columns=["artist", "title", "play_count"])
            continue
        df = rs.lyrics_long[rs.lyrics_long["word"].isin(tokens)]
        df = df.groupby("track_id", as_index=False, sort=False)["count"].max()
        df = df[df["count"] >= n]
        df = df.merge(tracks, on="track_id", how="left")
        df = df.merge(plays, on="song_id", how="left")
        df["play_count"] = df["play_count"].fillna(0)
        df = df.drop_duplicates("song_id")
        result[k] = _format_result(df, top_n)
    return result


def collection_word2vec(
    rs: MySpotifyRecommender,
    kw: list[str],
    n: int = 10,
    top_n: int = 50,
    neighbors: int = 10,
    model_name: str = "glove-wiki-gigaword-100",
) -> dict[str, pd.DataFrame]:
    """Keyword expanded with embedding neighbours; summed token count >= n."""
    model = gensim.downloader.load(model_name)
    vocab = set(rs.lyrics_long["word"].unique())
    tracks, plays = _tracks_plays(rs)

    result: dict[str, pd.DataFrame] = {}
    for k in kw:
        tokens: set[str] = set()
        seeds = [k] + ([w for w, _ in model.most_similar(k, topn=neighbors)] if k in model else [])
        for t in seeds:
            tokens.update(_vocab_tokens(t, vocab))
        if not tokens:
            result[k] = pd.DataFrame(columns=["artist", "title", "play_count"])
            continue
        df = rs.lyrics_long[rs.lyrics_long["word"].isin(sorted(tokens))]
        df = df.groupby("track_id", as_index=False, sort=False)["count"].sum()
        df = df[df["count"] >= n]
        df = df.merge(tracks, on="track_id", how="left")
        df = df.merge(plays, on="song_id", how="left")
        df["play_count"] = df["play_count"].fillna(0)
        result[k] = _format_result(df, top_n)
    return result


CLASSIFIERS = ("nb", "logistic", "sgd", "forest")


def _make_classifier(name: str, seed: int = 0, max_iter: int = 1000):
    if name == "nb":
        return MultinomialNB()
    if name == "logistic":
        return LogisticRegression(max_iter=max_iter)
    if name == "sgd":
        return SGDClassifier(loss="log_loss", max_iter=max_iter, random_state=seed)
    if name == "forest":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown classifier {name!r}; expected one of {CLASSIFIERS}")


def _lyric_counts(rs: MySpotifyRecommender):
    """Sparse track x word matrix from the tidy lyrics table + vocab + track ids."""
    df = rs.lyrics_long
    vocab = df["word"].astype("category").cat.categories.tolist()
    track_ids = df["track_id"].astype("category").cat.categories.tolist()
    trow = df["track_id"].astype("category").cat.codes.to_numpy()
    wcol = df["word"].astype("category").cat.codes.to_numpy()
    counts = sparse.coo_matrix(
        (df["count"].to_numpy().astype(np.float64), (trow, wcol)),
        shape=(len(track_ids), len(vocab)),
    ).tocsr()
    return counts, vocab, np.asarray(track_ids, dtype=object)


def _classify_keywords(
    counts,
    vocab: list[str],
    track_ids,
    tracks: pd.DataFrame,
    plays: pd.DataFrame,
    kw: list[str],
    classifier_names: tuple[str, ...],
    n: int,
    top_n: int,
    neg_ratio: int,
    max_iter: int,
    seed: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    empty = pd.DataFrame(columns=["artist", "title", "play_count"])
    result: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in classifier_names}
    vocab_set = set(vocab)
    rng = np.random.default_rng(seed)
    for k in kw:
        tokens = _vocab_tokens(k, vocab_set)
        if not tokens:
            for name in classifier_names:
                result[name][k] = empty.copy()
            continue
        cols = [vocab.index(t) for t in tokens]
        y = np.asarray(counts[:, cols].max(axis=1).toarray()).ravel() >= n
        y = y.astype(int)
        pos = np.flatnonzero(y == 1)
        if len(pos) == 0:
            for name in classifier_names:
                result[name][k] = empty.copy()
            continue
        train_idx, rest = train_test_split(
            np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
        )
        pos_train = train_idx[np.flatnonzero(y[train_idx] == 1)]
        neg_train_all = train_idx[np.flatnonzero(y[train_idx] == 0)]
        neg_train = rng.choice(
            neg_train_all,
            size=min(len(neg_train_all), len(pos_train) * neg_ratio),
            replace=False,
        )
        train_idx = np.concatenate([pos_train, neg_train])

        for name in classifier_names:
            pipe = Pipeline(
                [
                    ("tfidf", TfidfTransformer()),
                    ("clf", _make_classifier(name, seed, max_iter)),
                ]
            )
            pipe.fit(counts[train_idx], y[train_idx])
            score = pipe.predict_proba(counts[rest])[:, 1]
            df = pd.DataFrame({"track_id": track_ids[rest], "score": score})
            df = df.merge(tracks, on="track_id", how="left").merge(plays, on="song_id", how="left")
            df["play_count"] = df["play_count"].fillna(0)
            df = df.sort_values(["score", "play_count"], ascending=[False, False]).head(top_n)
            df.index = range(1, len(df) + 1)
            result[name][k] = df[["artist", "title", "play_count"]]
    return result


def collection_classification(
    rs: MySpotifyRecommender,
    kw: list[str],
    n: int = 5,
    top_n: int = 50,
    neg_ratio: int = 10,
    classifier: str = "logistic",
    max_iter: int = 1000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    counts, vocab, track_ids = _lyric_counts(rs)
    tracks, plays = _tracks_plays(rs)
    return _classify_keywords(
        counts, vocab, track_ids, tracks, plays, kw, (classifier,),
        n, top_n, neg_ratio, max_iter, seed,
    )[classifier]


def collection_classification_compare(
    rs: MySpotifyRecommender,
    kw: list[str],
    classifiers: tuple[str, ...] = CLASSIFIERS,
    n: int = 5,
    top_n: int = 50,
    neg_ratio: int = 10,
    max_iter: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, pd.DataFrame]]:
    counts, vocab, track_ids = _lyric_counts(rs)
    tracks, plays = _tracks_plays(rs)
    return _classify_keywords(
        counts, vocab, track_ids, tracks, plays, kw, classifiers,
        n, top_n, neg_ratio, max_iter, seed,
    )
