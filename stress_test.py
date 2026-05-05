from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

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
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print("-" * 50)

ask("What was net revenue in 2022?")
ask("How many employees does Molson Coors have?")
ask("Who are Molson Coors' main competitors?")
ask("How much did Molson Coors spend on marketing in 2023?")