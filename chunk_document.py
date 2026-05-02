from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader("molsoncoors.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(pages)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")
print(f"\nSample chunk:\n")
print(chunks[5].page_content)
