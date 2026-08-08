"""03 — Collections: 50 Songs About a Keyword (3 approaches)."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.loader import MySpotifyRecommender

try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ImportError:
    KeyedVectors = None
    api = None


def collection_baseline(
    rs: MySpotifyRecommender,
    keyword: str,
    n: int = 50,
    threshold: int = 1,
) -> pd.DataFrame:
    """Exact keyword match in lyrics, filter by threshold, rank by play count."""
    kw = keyword.lower()
    kw_lyrics = rs.lyrics_long[rs.lyrics_long["word"] == kw]
    tracks_with_kw = kw_lyrics[kw_lyrics["count"] >= threshold][
        ["track_id"]
    ].drop_duplicates()

    song_plays = rs.song_plays()
    enriched = rs.enrich(song_plays)

    matches = enriched.merge(tracks_with_kw, on="track_id", how="inner")
    result = (
        matches.sort_values("play_count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result.index += 1
    result.index.name = "rank"
    return result[["artist", "title", "play_count"]]


def collection_word2vec(
    rs: MySpotifyRecommender,
    keyword: str,
    n: int = 50,
    top_similar: int = 10,
) -> pd.DataFrame:
    """Expand keyword with semantically similar tokens via Word2Vec, combine lyric counts."""
    if KeyedVectors is None:
        raise ImportError(
            "gensim is required for word2vec. Install with: uv sync --extra word2vec"
        )

    try:
        wv = KeyedVectors.load_word2vec_format(
            "GoogleNews-vectors-negative300.bin", binary=True
        )
    except (FileNotFoundError, OSError):
        print("GoogleNews word2vec not available -- using gensim's built-in small model")
        wv = api.load("glove-wiki-gigaword-100")

    kw = keyword.lower()
    try:
        similar_words = [w for w, _ in wv.most_similar(kw, topn=top_similar)]
    except KeyError:
        similar_words = []

    words = [kw] + similar_words
    print(f"  Expanded '{keyword}' -> {words}")

    kw_lyrics = rs.lyrics_long[rs.lyrics_long["word"].isin(words)]
    track_scores = (
        kw_lyrics.groupby("track_id", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "kw_score"})
    )

    song_plays = rs.song_plays()
    enriched = rs.enrich(song_plays)
    merged = enriched.merge(track_scores, on="track_id", how="inner")

    result = (
        merged.sort_values("play_count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result.index += 1
    result.index.name = "rank"
    return result[["artist", "title", "play_count"]]


def collection_classification(
    rs: MySpotifyRecommender,
    keyword: str,
    n: int = 50,
) -> pd.DataFrame:
    """Train a classifier on labelled data, predict scores for all tracks."""
    kw = keyword.lower()

    kw_tracks = (
        rs.lyrics_long[rs.lyrics_long["word"] == kw][["track_id"]].drop_duplicates()
    )
    all_tracks = rs.lyrics_long[["track_id"]].drop_duplicates()

    labelled = all_tracks.copy()
    labelled["label"] = labelled["track_id"].isin(kw_tracks["track_id"]).astype(int)

    track_bow = (
        rs.lyrics_long.groupby("track_id")
        .apply(lambda g: " ".join(g["word"].tolist()), include_groups=False)
        .reset_index(name="text")
    )
    labelled = labelled.merge(track_bow, on="track_id", how="inner")

    if len(labelled) > 200_000:
        labelled = labelled.sample(200_000, random_state=42)

    X = labelled["text"]
    y = labelled["label"]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10_000, stop_words="english")),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    print(
        f"  Trained classifier on {len(labelled)} tracks "
        f"(pos={y.sum()}, neg={len(y) - y.sum()})"
    )

    all_bow = track_bow.copy()
    all_bow["score"] = pipe.predict_proba(all_bow["text"])[:, 1]

    song_plays = rs.song_plays()
    enriched = rs.enrich(song_plays)
    merged = enriched.merge(all_bow[["track_id", "score"]], on="track_id", how="inner")

    result = (
        merged.sort_values("play_count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result.index += 1
    result.index.name = "rank"
    return result[["artist", "title", "play_count"]]
