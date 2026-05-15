import os
import mlflow

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("barebones_rag_demo")
mlflow.langchain.autolog()

docs = [
    Document(page_content="Alpha Company handles healthcare benefits, claims, eligibility, and member support."),
    Document(page_content="Call volume forecasting helps staffing teams predict busy days in claims call centers."),
    Document(page_content="Random forests can forecast volume using lag features, rolling averages, and calendar features."),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="claims_demo_docs",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer using only this context:

{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

with mlflow.start_run(run_name="tiny_langchain_rag_test"):
    question = "How could call volume forecasting help a benefits company?"

    response = rag_chain.invoke(question)

    mlflow.log_param("question", question)
    mlflow.log_text(response.content, "rag_answer.txt")

    print(response.content)