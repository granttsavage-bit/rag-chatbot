from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

loader = PyPDFLoader("MolsonCoors.pdf")
pages = loader.load()

print(f"Total pages loaded: {len(pages)}")
print(f"\nSample text from page 1:\n")
print(pages[0].page_content[:500])
