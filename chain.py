from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import ollama
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


def get_retrieval_chain():
    embeddings = OllamaEmbeddings(model="mistral")
    llm = ChatOllama(model="mistral")
    db = FAISS.load_local("faiss_index", embeddings)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant that help people understand data from google review dataset.
        All data in context are google review on spotify apps.
        Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        All questions must be supported by facts in the context
        All reasoning must be done step by step.
        Explain the reasoning.
        When looking at multiple rows, explain the reasoning for each row one by one.

        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    return create_retrieval_chain(retriever, document_chain)
