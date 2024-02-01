from chain import get_chain, get_retrieval_qa_chain

import streamlit as st

from langchain.globals import set_debug

set_debug(False)


def main():
    st.title("Spotify Review Chatbot")

    chain = get_retrieval_qa_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for response in chain.stream(
                {"question": prompt, "chat_history": st.session_state.chat_history[-5:]}
            ):
                full_response += response or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.chat_history.append({"human": prompt, "ai": full_response})


if __name__ == "__main__":
    main()
