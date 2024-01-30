import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Remove empty review text
    df = df.dropna(subset=["review_text"])
    # Remove {} in author_name and review_text
    df["author_name"] = df["author_name"].apply(
        lambda x: x.replace("{", "").replace("}", "")
    )
    df["review_text"] = df["review_text"].apply(
        lambda x: x.replace("{", "").replace("}", "")
    )
    return df
