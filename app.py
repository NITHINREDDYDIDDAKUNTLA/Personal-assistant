from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("PERSONAL ASSISTANT")

if "memory"not in st.session_state:
    st.session_state.memory=ConversationBufferWindowMemory(k=2)
model=ChatGroq(model="openai/gpt-oss-120b")
chain=ConversationChain(llm=model,memory=st.session_state.memory)

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt=st.chat_input("Enter your prompt:")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    result=chain.invoke({"input":prompt})
    with st.chat_message("assistant"):
        st.write(result["response"])
    st.session_state.messages.append({"role":"assistant","content":result["response"]})
