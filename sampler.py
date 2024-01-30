"""
Use this code to generate sample data from SPOTIFY_REVIEWS.csv
"""

import pandas as pd
from helper import preprocess


def sampler(path_str: str):
    df = pd.read_csv(path_str)
    df = preprocess(df)

    df_sampled = df.groupby("review_rating", group_keys=False).apply(
        lambda x: x.sample(frac=0.0001, random_state=2024)
    )
    df_sampled.to_csv(f"SPOTIFY_REVIEWS_SAMPLE.csv", header=True, index=False)


if __name__ == "__main__":
    sampler("SPOTIFY_REVIEWS.csv")
