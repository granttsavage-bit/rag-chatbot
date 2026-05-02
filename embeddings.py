from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

# Load and chunk the document
loader = PyPDFLoader("molsoncoors.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)

# Create embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Test on a single chunk first
sample_embedding = embeddings.embed_query(chunks[0].page_content)

print(f"Total chunks to embed: {len(chunks)}")
print(f"Embedding dimensions: {len(sample_embedding)}")
print(f"\nFirst 10 values of the embedding vector:")
print(sample_embedding[:10])