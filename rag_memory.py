from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
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

# Store chat history manually
chat_history = []

# Prompt with memory
prompt = PromptTemplate.from_template("""You are a financial analyst assistant helping users 
understand the Molson Coors 2023 annual report (10-K).

Use the following context from the document and the chat history to answer the question.
If the answer is not in the context, say "I don't have enough information in the document to answer that."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""")

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    if not history:
        return "No previous conversation."
    return "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])

# Chat function
def chat(question):
    context = format_docs(retriever.invoke(question))
    history = format_history(chat_history)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
        "chat_history": history
    })

    chat_history.append((question, answer))
    return answer

# Test multi-turn conversation
questions = [
    "What was Molson Coors net revenue in 2023?",
    "How does that compare to 2022?",
    "What drove the change?",
    "Who is responsible for the company's financial performance?"
]

for question in questions:
    print(f"\nQ: {question}")
    print(f"A: {chat(question)}")
    print("-" * 50)