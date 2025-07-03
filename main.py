# Medical Diagnosis Assistant - RAG-based Agent

# Install required packages
# pip install langchain openai biopython faiss-cpu tiktoken

import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from Bio import Entrez
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Set environment
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
Entrez.email = "your_email@example.com"

# Step 1: PubMed search tool
def search_pubmed(query, max_results=5):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    papers = []
    if ids:
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts = handle.read().split("\n\n")
        for i, abs_text in enumerate(abstracts):
            papers.append(Document(page_content=abs_text.strip(), metadata={"source": ids[i]}))
    return papers

# Step 2: Build the vector store from PubMed abstracts
def build_retriever(query):
    docs = search_pubmed(query)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

# Step 3: Create a QA chain
def build_qa_chain(symptom_input):
    retriever = build_retriever(symptom_input)
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever, return_source_documents=True)
    return qa_chain

# Step 4: Custom diagnostic tool class
class MedicalDiagnosisTool(BaseTool):
    name = "Medical Diagnosis Expert"
    description = "Provides a diagnosis suggestion based on user symptoms and medical literature"

    def _run(self, query: str):
        chain = build_qa_chain(query)
        response = chain.run(query)
        return response

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported")

# Step 5: Initialize agent
llm = ChatOpenAI(temperature=0.2)
medical_tool = MedicalDiagnosisTool()
tools = [medical_tool]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Step 6: Sample run
if __name__ == "__main__":
    user_input = "fever, rash, joint pain"
    print("\nDiagnosing for symptoms:", user_input)
    response = agent.run(f"Given the symptoms: {user_input}, what are possible diagnoses and how should a doctor proceed?")
    print("\nDiagnosis Suggestion:\n", response)
