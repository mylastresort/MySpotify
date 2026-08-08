import pandas as pd

from src.data.loader import MySpotifyRecommender


def top_n_songs(rs: MySpotifyRecommender, n: int = 10, user_id: str | None = None) -> pd.DataFrame:
    if user_id is not None:
        # (user_id, song_id) is unique -> no groupby needed
        top = rs.triplets[rs.triplets["user_id"] == user_id][["song_id", "play_count"]] \
            .nlargest(n, "play_count")
    else:
        top = rs.triplets[["song_id", "play_count"]] \
            .groupby("song_id", as_index=False)["play_count"].sum() \
            .nlargest(n, "play_count")

    df = top.merge(
        rs.tracks.drop_duplicates("song_id")[["song_id", "track_id", "artist", "title"]],
        on="song_id",
        how="left",
    )
    # order columns
    df = df[["track_id", "artist", "title", "play_count"]]
    return df