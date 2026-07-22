import streamlit as st

st.set_page_config(page_title="Project Aria Chatbot", page_icon="💬", layout="centered")

st.title("💬 Project Aria Chatbox")
st.caption("A simple Streamlit chat UI you can expand with your own chatbot logic.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the Project Aria workspace."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "hello" in prompt.lower():
        reply = "Hello! I’m your Streamlit chat demo."
    elif "help" in prompt.lower():
        reply = "You can customize this UI and connect it to an LLM or your own backend."
    else:
        reply = "Thanks for your message. This is a starter chatbox UI for Project Aria."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the Project Aria workspace."}
    ]
    st.rerun()
