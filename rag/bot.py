import os
from openai import OpenAI
from dotenv import load_dotenv
from rag.retriever import retrieve_context

# Access the API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
# The client automatically picks up the key if the env var OPENAI_API_KEY is set
load_dotenv()
client = OpenAI(api_key=openai_api_key)
#load_dotenv()
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_bot(question):

    context = retrieve_context(question)

    system_prompt = """
    You are a University Parent Helpdesk Assistant.
    Answer ONLY from the provided context.
    If answer is not found, say:
    "Please contact university administration."
    Keep answers short and polite.
    """

    user_prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )

    return response.choices[0].message.content
