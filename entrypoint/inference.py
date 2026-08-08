"""Inference — generate recommendations from trained models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import MySpotifyRecommender
from src.models.collaborative_filtering import (
    build_user_item_matrix,
    recommend_tracks_df,
    recommend_users_df,
    train_test_split,
)
from src.models.collections import collection_baseline, collection_classification
from src.models.spotify_features import (
    because_you_listened_to,
    build_artist_user_matrix,
    fans_also_like,
    your_genre_mix,
)
from src.models.top_n import top_n_songs
from src.models.top_n_genre import top_n_per_genre


def main() -> None:
    print("[inference] Loading data...")
    rs = MySpotifyRecommender.from_files()

    # Top 250
    print("\n=== Top 250 Most-Played Songs ===")
    top250 = top_n_songs(rs, n=250)
    print(top250.head(10))

    # Top 100 by genre
    print("\n=== Top 100 Rock Songs ===")
    top100_rock = top_n_per_genre(rs, "Rock", n=100)
    print(top100_rock.head(10))

    # Collections
    print("\n=== Collections: Songs About 'love' ===")
    baseline_love = collection_baseline(rs, "love", n=50)
    print(baseline_love.head(10))

    clf_love = collection_classification(rs, "love", n=50)
    print(clf_love.head(10))

    # Collaborative Filtering
    print("\n=== Collaborative Filtering ===")
    train, test = train_test_split(rs, test_ratio=0.2)
    user_item, user_idx, song_idx, idx_song = build_user_item_matrix(train)

    sample_user = (
        train.groupby("user_id")["play_count"]
        .count()
        .sort_values(ascending=False)
        .index[0]
    )
    print(f"\nUser-based CF for {sample_user}:")
    recs = recommend_users_df(sample_user, user_item, user_idx, idx_song, rs.tracks)
    print(recs)

    # Spotify-inspired features
    print("\n=== Because You Listened To ===")
    recs_bs, seed_artist = because_you_listened_to(rs, sample_user)
    print(f"Seed artist: {seed_artist}")
    print(recs_bs)

    print("\n=== Your Genre Mix ===")
    recs_gm, genre = your_genre_mix(rs, sample_user)
    print(f"Genre: {genre}")
    print(recs_gm)

    print("\n=== Fans Also Like ===")
    artist_matrix, artist_idx, idx_artist = build_artist_user_matrix(rs)
    recs_fa = fans_also_like(rs, seed_artist, artist_matrix, artist_idx, idx_artist)
    print(recs_fa)

    print("\n[inference] Done.")


if __name__ == "__main__":
    main()
