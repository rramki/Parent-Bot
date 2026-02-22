import streamlit as st
from rag.bot import ask_bot

st.title("🎓 University Parent Helpdesk")

question = st.text_input("Ask your question:")

if st.button("Submit"):
    if question:
        answer = ask_bot(question)
        st.write("### Answer:")
        st.write(answer)
