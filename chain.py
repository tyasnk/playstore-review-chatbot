from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough
)
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage

import pandas as pd
import os
from pydantic import BaseModel
from typing import Optional, List, Dict, Sequence
from operator import itemgetter


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def get_chain():
    """
    Simple chain with reviews data used directly in prompt

    """
    llm = ChatOllama(model="mistral", temperature=0)

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

    """

    prompt = ChatPromptTemplate.from_template(
        prompt_template + "\nQuestion: {question}"
    )
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


def get_retrieval_qa_chain():
    """
    Use RAG with Faiss with chat history

    """
    embeddings = OllamaEmbeddings(model="mistral", temperature=0)
    llm = ChatOllama(model="mistral")
    db = FAISS.load_local("faiss_index", embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 200})

    output_parser = StrOutputParser()

    RESPONSE_TEMPLATE = """
    You are an assistant that help people understand data from google review dataset.
    All data in context are google review on spotify apps.
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    All questions must be supported by facts in the context
    All reasoning must be done step by step.

    Question: {question}
    """

    REPHRASE_TEMPLATE = """\
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone Question:
    """

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = CONDENSE_QUESTION_PROMPT | llm | output_parser
    conversation_chain = condense_question_chain | retriever
    retriever_chain = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))),
            conversation_chain,
        ),
        (RunnableLambda(itemgetter("question")) | retriever),
    )

    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = prompt | llm | output_parser
    return (
        {
            "question": RunnableLambda(itemgetter("question")),
            "chat_history": RunnableLambda(serialize_history),
        }
        | _context
        | response_synthesizer
    )

# TODO:
# 1. The solution above has limitation on prompt context length so it decrease the accuracy. For large data, we need to try another approach such as using agents or function calling so we can process the whole dataset
# 2. We use mistral:7b which is free and open source llm so the accuracy is not so well. We need to try using gpt3.5 or gpt4 (but i have no money :( )
# 3. Need to try to improve the prompt for better accuracy
