"""Train pipeline — build CSVs from raw data and evaluate models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure src/ is importable when running from project root or entrypoint/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import MySpotifyRecommender
from src.models.collaborative_filtering import (
    build_user_item_matrix,
    evaluate_item_cf,
    evaluate_user_cf,
    train_test_split,
)
from src.models.top_n import top_n_songs
from src.models.top_n_genre import top_n_per_genre


def load_config(env: str = "local") -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config" / f"{env}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(env: str = "local") -> None:
    cfg = load_config(env)
    print(f"[train] Config: {env}")
    print(f"[train] Project: {cfg['project']['name']} v{cfg['project']['version']}")

    # 1. Load data
    print("\n--- Loading data ---")
    rs = MySpotifyRecommender.from_files()

    # 2. Top N baseline
    print("\n--- Top 250 songs ---")
    top250 = top_n_songs(rs, n=cfg["model"]["top_n"]["default_top"])
    print(top250.head(10))

    # 3. Genre baseline
    print("\n--- Top 100 Rock songs ---")
    top100_rock = top_n_per_genre(rs, "Rock", n=100)
    print(top100_rock.head(10))

    # 4. Collaborative Filtering
    print("\n--- Collaborative Filtering ---")
    cf_cfg = cfg["model"]["collaborative_filtering"]
    train, test = train_test_split(rs, test_ratio=cf_cfg["test_ratio"])

    user_item, user_idx, song_idx, idx_song = build_user_item_matrix(train)

    avg_pk_user = evaluate_user_cf(
        rs, train, test, user_item, user_idx, idx_song
    )
    avg_pk_item = evaluate_item_cf(
        rs, train, test, user_item, song_idx, idx_song
    )

    print(f"\n{'Method':<30s} {'Precision@10':>12s}  {'Status':>6s}")
    print("-" * 52)
    print(f"{'User-based CF':<30s} {avg_pk_user:>11.4f}   {'PASS' if avg_pk_user > cf_cfg['precision_target'] else 'FAIL':>6s}")
    print(f"{'Item-based CF':<30s} {avg_pk_item:>11.4f}   {'PASS' if avg_pk_item > cf_cfg['precision_target'] else 'FAIL':>6s}")

    print("\n[train] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MySpotify train pipeline")
    parser.add_argument("--env", default="local", choices=["local", "prod"])
    args = parser.parse_args()
    main(env=args.env)
