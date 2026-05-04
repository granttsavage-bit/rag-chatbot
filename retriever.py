from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

# Load the existing vectorstore from disk
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Search function
def search(question, k=3):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    results = vectorstore.similarity_search(question, k=k)
    for i, doc in enumerate(results):
        print(f"\nChunk {i+1} (page {doc.metadata.get('page', 'unknown') + 1}):")
        print(doc.page_content[:400])
        print()

# Test with 5 real questions
search("What was Molson Coors net revenue in 2023?")
search("What are the main risks facing the company?")
search("Who is the CEO of Molson Coors?")
search("What brands does Molson Coors own?")
search("What is the company's strategy for growth?")
