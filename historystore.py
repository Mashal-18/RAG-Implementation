import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json

# Step 1: Load and process the PDF
def process_pdf(C:\Users\hp\Desktop\New folder\paper01.pdf):
    print("Loading and processing the PDF...")
    loader = PyPDFLoader(C:\Users\hp\Desktop\New folder\paper01.pdf)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Step 2: Create a vectorstore for retrieval
def create_vectorstore(texts):
    print("Creating embeddings and setting up vectorstore...")
    embeddings = HuggingFaceEmbeddings()  # Use HuggingFace embeddings instead of OpenAI
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Step 3: Initialize the Retrieval-Augmented Generation (RAG) model
def setup_qa_chain(vectorstore):
    print("Setting up QA chain...")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=None, retriever=retriever)  # LLM setup placeholder
    return qa_chain

# Step 4: Interactive question-answering with session history
def interactive_qa(qa_chain):
    session_history = []
    print("You can now start asking questions. Type 'exit' to end the session.")
    
    while True:
        question = input("Your question: ")
        if question.lower() == 'exit':
            break
        
        # Query the QA chain
        answer = qa_chain.run(question)
        
        # Save the question and answer to session history
        session_history.append({"question": question, "answer": answer})
        
        # Display the answer
        print(f"Answer: {answer}\n")

    # Save the session history to a file
    with open("session_history.json", "w") as f:
        json.dump(session_history, f, indent=4)
    print("Session history saved to 'session_history.json'.")

if __name__ == "__main__":
    # Step 1: Provide the PDF file path
    pdf_path = input("Enter the path to your PDF file: ")
    texts = process_pdf(C:\Users\hp\Desktop\New folder\paper01.pdf)

    # Step 2: Create a vectorstore
    vectorstore = create_vectorstore(texts)

    # Step 3: Set up the QA chain
    qa_chain = setup_qa_chain(vectorstore)

    # Step 4: Run the interactive QA session
    interactive_qa(qa_chain)
