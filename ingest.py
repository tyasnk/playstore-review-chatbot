from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import pandas as pd

from langchain.docstore.document import Document


def ingest(path_str: str):
    columns_to_embed = [
        "review_text",
        "review_rating",
        "review_likes",
        "author_app_version",
        "review_timestamp",
    ]
    columns_to_metadata = ["review_id", "pseudo_author_id", "author_name"]

    documents = []
    df = pd.read_csv(path_str)
    for _, row in df.iterrows():
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(
            f"{k.strip()}: {str(v).strip()}" for k, v in values_to_embed.items()
        )
        documents.append(Document(page_content=to_embed, metadata=to_metadata))

    embedding = OllamaEmbeddings(model="mistral")

    db = FAISS.from_documents(documents, embedding)
    db.save_local("faiss_index")


if __name__ == "__main__":
    ingest("SPOTIFY_REVIEWS_SAMPLE.csv")
