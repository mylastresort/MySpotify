"""MySpotify — entry point for CLI usage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import MySpotifyRecommender
from src.models.top_n import top_n_songs


def main() -> None:
    rs = MySpotifyRecommender.from_files()
    top = top_n_songs(rs, n=10)
    print(top)


if __name__ == "__main__":
    main()
