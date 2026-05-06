import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Molson Coors 10-K Chatbot",
    page_icon="🍺",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.title("About This App")
    st.markdown("""
    This chatbot answers questions about the **Molson Coors 2023 Annual Report (10-K)** 
    filed with the SEC on February 20, 2024.
    
    **How it works:**
    - Document is split into 716 chunks
    - Each chunk is stored as a vector embedding in ChromaDB
    - Your question is matched to the most relevant chunks
    - GPT-3.5 synthesizes an answer from those chunks
    
    **Tech Stack:**
    - Python
    - LangChain
    - OpenAI API
    - ChromaDB
    - Streamlit
    
    **Built by:** Grant Savage
    """)
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🍺 Molson Coors 10-K Chatbot")
st.caption("Ask any question about the Molson Coors 2023 Annual Report")

# Load or build vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    pdf_path = os.path.join(os.path.dirname(__file__), "molsoncoors.pdf")
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")

    if not os.path.exists(chroma_path):
        st.info("Building knowledge base for the first time... this may take a minute.")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_path
        )
    else:
        vectorstore = Chroma(
            persist_directory=chroma_path,
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

def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    sources = sorted(set([doc.metadata.get("page", 0) + 1 for doc in docs]))
    return answer, sources

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            st.info(f"📄 Sources: Pages {', '.join(map(str, message['sources']))}")

# Chat input
if question := st.chat_input("Ask a question about the 10-K..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            answer, sources = ask(question)
        st.markdown(answer)
        st.info(f"📄 Sources: Pages {', '.join(map(str, sources))}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })