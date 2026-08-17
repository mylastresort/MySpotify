"""Collections: 50 most-played songs about a keyword (baseline, word2vec, classification)."""

import gensim.downloader
import numpy as np
import polars as pl
from gensim.parsing.preprocessing import PorterStemmer
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data import MySpotifyRecommender

_STEMMER = PorterStemmer()
EMPTY = pl.DataFrame(
    schema={"artist": pl.Utf8, "title": pl.Utf8, "play_count": pl.Int64}
)


def _vocab_tokens(kw: str, vocab: set[str]) -> list[str]:
    if kw in vocab:
        return [kw]
    stem = _STEMMER.stem(kw)
    return [stem] if stem in vocab else []


def _tracks_plays(rs: MySpotifyRecommender):
    tracks = rs.tracks.unique(subset="track_id", keep="first").select(
        "track_id", "song_id", "artist", "title"
    )
    plays = rs.triplets.group_by("song_id").agg(pl.col("play_count").sum())
    return tracks, plays


def _format_result(df: pl.DataFrame, top_n: int) -> pl.DataFrame:
    return (
        df.sort("play_count", descending=True, maintain_order=True)
        .head(top_n)
        .with_row_index("index", offset=1)
        .select("index", "artist", "title", "play_count")
    )


def collection_baseline(
    rs: MySpotifyRecommender,
    kw: list[str],
    n: int = 10,
    top_n: int = 50,
) -> dict[str, pl.DataFrame]:
    """Tracks where the keyword token appears at least ``n`` times."""
    vocab = set(rs.lyrics_long["word"].unique().to_list())
    tracks, plays = _tracks_plays(rs)

    result: dict[str, pl.DataFrame] = {}
    for k in kw:
        tokens = _vocab_tokens(k, vocab)
        if not tokens:
            result[k] = EMPTY
            continue
        df = (
            rs.lyrics_long.filter(pl.col("word").is_in(tokens))
            .filter(pl.col("count") >= n)
            .join(tracks, on="track_id", how="left")
            .join(plays, on="song_id", how="left")
            .with_columns(pl.col("play_count").fill_null(0))
            .unique(subset="song_id", keep="first")
        )
        result[k] = _format_result(df, top_n)
    return result


def collection_word2vec(
    rs: MySpotifyRecommender,
    kw: list[str],
    n: int = 10,
    top_n: int = 50,
    neighbors: int = 10,
    model_name: str = "glove-wiki-gigaword-100",
) -> dict[str, pl.DataFrame]:
    """Keyword expanded with embedding neighbours; summed token count >= n."""
    model = gensim.downloader.load(model_name)
    vocab = set(rs.lyrics_long["word"].unique().to_list())
    tracks, plays = _tracks_plays(rs)

    result: dict[str, pl.DataFrame] = {}
    for k in kw:
        seeds = [k] + ([w for w, _ in model.most_similar(k, topn=neighbors)] if k in model else [])
        tokens = set()
        for t in seeds:
            tokens.update(_vocab_tokens(t, vocab))
        if not tokens:
            result[k] = EMPTY
            continue
        df = (
            rs.lyrics_long.filter(pl.col("word").is_in(sorted(tokens)))
            .filter(pl.col("count") >= n)
            .join(tracks, on="track_id", how="left")
            .join(plays, on="song_id", how="left")
            .with_columns(pl.col("play_count").fill_null(0))
        )
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
    df = rs.lyrics_long
    cat = df.select(
        pl.col("track_id").cast(pl.Categorical).alias("tid"),
        pl.col("word").cast(pl.Categorical).alias("word"),
    )
    vocab = cat["word"].cat.get_categories().to_list()
    track_ids = cat["tid"].cat.get_categories().to_list()
    counts = sparse.coo_matrix(
        (
            df["count"].to_numpy().astype(np.float64),
            (cat["tid"].to_physical().to_numpy(), cat["word"].to_physical().to_numpy()),
        ),
        shape=(len(track_ids), len(vocab)),
    ).tocsr()
    return counts, vocab, np.asarray(track_ids, dtype=object)


def _classify_keywords(
    counts,
    vocab: list[str],
    track_ids,
    tracks: pl.DataFrame,
    plays: pl.DataFrame,
    kw: list[str],
    classifier_names: tuple[str, ...],
    n: int,
    top_n: int,
    neg_ratio: int,
    max_iter: int,
    seed: int,
) -> dict[str, dict[str, pl.DataFrame]]:
    result: dict[str, dict[str, pl.DataFrame]] = {name: {} for name in classifier_names}
    vocab_set = set(vocab)
    rng = np.random.default_rng(seed)
    for k in kw:
        tokens = _vocab_tokens(k, vocab_set)
        if not tokens:
            for name in classifier_names:
                result[name][k] = EMPTY
            continue
        cols = [vocab.index(t) for t in tokens]
        y = np.asarray(counts[:, cols].max(axis=1).toarray()).ravel() >= n
        y = y.astype(int)
        pos = np.flatnonzero(y == 1)
        if len(pos) == 0:
            for name in classifier_names:
                result[name][k] = EMPTY
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
            df = (
                pl.DataFrame({"track_id": track_ids[rest].tolist(), "score": score})
                .join(tracks, on="track_id", how="left")
                .join(plays, on="song_id", how="left")
                .with_columns(pl.col("play_count").fill_null(0))
                .sort(["score", "play_count"], descending=[True, False])
                .head(top_n)
                .with_row_index("index", offset=1)
                .select("index", "artist", "title", "play_count")
            )
            result[name][k] = df
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
) -> dict[str, pl.DataFrame]:
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
) -> dict[str, dict[str, pl.DataFrame]]:
    counts, vocab, track_ids = _lyric_counts(rs)
    tracks, plays = _tracks_plays(rs)
    return _classify_keywords(
        counts, vocab, track_ids, tracks, plays, kw, classifiers,
        n, top_n, neg_ratio, max_iter, seed,
    )
