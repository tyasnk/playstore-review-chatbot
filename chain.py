from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.utilities.python import PythonREPL
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chains import create_retrieval_chain

import pandas as pd
import os


def get_chain():
    """
    Simple chain with reviews data used directly in prompt

    """
    llm = ChatOllama(model="mistral", temperature=0.1)

    df = pd.read_csv("SPOTIFY_REVIEWS_SAMPLE.csv")
    df = df[
        [
            "author_name",
            "review_text",
            "review_rating",
            "review_likes",
            "author_app_version",
            "review_timestamp",
        ]
    ]
    df_str = df.to_markdown(index=False)

    prompt_template = f"""

    You are an assistant that help people understand data from google review dataset.
    All data in context are google review on spotify apps.
    Answer the following question based only on the provided context:

    <context>
    {df_str}
    </context>

    All questions must be supported by facts in the context
    All reasoning must be done step by step.
    Explain the reasoning.

    """

    prompt = ChatPromptTemplate.from_template(prompt_template + "\nQuestion: {input}")
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


def get_retrieval_chain():
    """
    Use RAG with Faiss to get context from embedded document

    """
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

    retriever = db.as_retriever(search_kwargs={"k": 10})
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    )

    output_parser = StrOutputParser()

    return setup_and_retrieval | prompt | llm | output_parser


def get_map_reduce_chain(question: str):
    """
    Get answer from multiple batch and then summarize the result
    """
    llm = ChatOllama(model="mistral", temperature=0.1)
    output_parser = StrOutputParser()

    df = pd.read_csv("SPOTIFY_REVIEWS_SAMPLE.csv")
    df = df[
        [
            "author_name",
            "review_text",
            "review_rating",
            "review_likes",
            "author_app_version",
            "review_timestamp",
        ]
    ]

    df = df.dropna(subset=["review_text"])

    n = 30  # chunk row size
    list_df = [df[i : i + n] for i in range(0, df.shape[0], n)]

    map_prompt_template = """

    You are an assistant that help people understand data from google review dataset.
    All data in context are google review on spotify apps.
    Answer the following question based only on the provided context:

    <context>
    {df_str}
    </context>

    All questions must be supported by facts in the context
    All reasoning must be done step by step.
    Explain the reasoning.
    The answer must be no more than 3 sentence

    """

    summary = ""

    print("start batch")
    for b_df in list_df:
        df_str = b_df.to_markdown(index=False)
        map_prompt = map_prompt_template.format(df_str=df_str)
        map_chain_prompt = ChatPromptTemplate.from_template(
            map_prompt + "\nQuestion: {input}"
        )
        map_chain = map_chain_prompt | llm | output_parser
        map_result = map_chain.invoke({"input": question})
        print(map_result)
        summary += f"\n{map_result}"

    reduce_prompt_template = f"""
    You are an assistant that help people understand data from google review dataset.
    All data in context are summarization from same question with different batch data before.
    Answer the following question based only on the provided context:

    <context>
    {summary}
    </context>

    the answer should be summarization from context that answer the question

    """

    print(f"final prompt : {reduce_prompt_template}")

    prompt = ChatPromptTemplate.from_template(
        reduce_prompt_template + "\nQuestion: {input}"
    )
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


def get_table_agent():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, api_key=openai_api_key)

    df = pd.read_csv("SPOTIFY_REVIEWS_SAMPLE.csv")
    df_columns = df.columns.to_list()

    python_repl = PythonREPL(locals={"df": df})
    # You can create the tool to pass to an agent
    tools = [
        Tool(
            name="Find Positive Feedback",
            description=f"""
        Useful for when you need to answer questions about positive feedback stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer. 
        'df' has the following columns: {df_columns}

        <user>: What are the specific features or aspects that users appreciate the most in our application?
        <assistant>: df[df['review_rating'] >= 4]["review_text"]
        <assistant>: I Need to Find best relevant keyword based on review_text
        """,
            func=python_repl.run,
        )
    ]

    agent_kwargs = {
        "prefix": f"""You are an assistant that help people understand data from google review dataset.
    All data in context are google review on spotify apps.
    All questions must be supported by facts in the context
    All reasoning must be done step by step.
    Explain the reasoning."""
    }

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )

    return agent
