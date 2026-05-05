from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Load vectorstore
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt template
prompt = PromptTemplate.from_template("""You are a financial analyst assistant helping users 
understand the Molson Coors 2023 annual report (10-K).

Use the following context from the document to answer the question. 
If the answer is not in the context, say "I don't have enough information 
in the document to answer that."

Context:
{context}

Question: {question}

Answer:""")

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Helper to format retrieved chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test it
questions = [
    "What was Molson Coors net revenue in 2023?",
    "Who is the CEO of Molson Coors?",
    "What brands does Molson Coors own?",
    "What are the main risks facing the company?"
]

for question in questions:
    print(f"\nQ: {question}")
    print(f"A: {chain.invoke(question)}")
    print("-" * 50)