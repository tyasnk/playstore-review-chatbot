from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
import pandas as pd


def ingest(path_str: str):
    df = pd.read_csv(path_str)

    def _table_to_text(x):
        return f'The text review is "{x["review_text"]}". The review rating is {x["review_rating"]}. The review likes is {x["review_likes"]}. The review timestamp is {x["review_timestamp"]}'

    df["table_to_text"] = df.apply(lambda x: _table_to_text(x), axis=1)

    embedding = OllamaEmbeddings(model="mistral")

    loader = DataFrameLoader(df, page_content_column="table_to_text")
    documents = loader.load()

    db = FAISS.from_documents(documents, embedding)
    db.save_local("faiss_index")


if __name__ == "__main__":
    ingest("SPOTIFY_REVIEWS_SAMPLE.csv")
