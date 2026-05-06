import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

st.set_page_config(
    page_title="Molson Coors 10-K Chatbot",
    page_icon="🍺",
    layout="centered"
)

st.title("🍺 Molson Coors 10-K Chatbot")
st.caption("Ask any question about the Molson Coors 2023 Annual Report")

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 7})

retriever = load_vectorstore()

prompt = PromptTemplate.from_template("""You are a financial analyst assistant helping users 
understand the Molson Coors 2023 annual report (10-K).

Use the following context from the document to answer the question.
If the answer is not in the context, say "I don't have enough information in the document to answer that."
Do not perform calculations — only report numbers directly stated in the document.
If you find a segment number but not a total, say so explicitly.
Always try to name specific entities rather than giving general descriptions.
Do not make up numbers or facts.

Context:
{context}

Question: {question}

Answer:""")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask(question):
    context = format_docs(retriever.invoke(question))
    return (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a question about the 10-K..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            answer = ask(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})