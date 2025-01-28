import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain.chains import create_retrieval_chain  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain.agents import Tool  # for integrating tools

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please configure it in your .env file.")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt templates for simple and complex queries
simple_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context only. Be concise and to the point.
<context>
{context}
</context>
Question: {input}
""")

complex_prompt = ChatPromptTemplate.from_template("""
Provide a detailed answer based on the context below. Include explanations and analysis if applicable.
<context>
{context}
</context>
Question: {input}
""")

# Tool for Document Extraction from PDF
def extract_pdf_content(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    text_documents = loader.load()
    print(f"Extracted {len(text_documents)} documents from the PDF.")
    return text_documents

# Tool for Classifying Questions
def classify_question(query):
    """Classifies the question as simple or complex."""
    keywords = ["explain", "analyze", "detailed", "complex"]
    if len(query.split()) > 12 or any(keyword in query.lower() for keyword in keywords):
        return "complex"
    return "simple"

# Tool for Creating a Vector Store from Text
def create_vector_db_from_text(text_documents):
    """Creates a vector store from the extracted text."""
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    document_chunks = text_splitter.split_documents(text_documents)
    print(f"Number of document chunks: {len(document_chunks)}")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},  # Change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create a vector store
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    print("Vector store created successfully.")
    return vector_store

# Tool for Retrieving the Answer
def get_answer_from_query(query, vector_store):
    """Answers a query using the appropriate agent."""
    if not vector_store:
        return "Vector store is empty. Ensure the PDF is properly processed."

    # Choose the appropriate prompt template based on question complexity
    agent_type = "simple" if classify_question(query) == "simple" else "complex"
    print(f"Switching to {agent_type} agent to answer your query.")

    if agent_type == "complex":
        document_chain = create_stuff_documents_chain(llm, complex_prompt)
    else:
        document_chain = create_stuff_documents_chain(llm, simple_prompt)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print(f"Retrieving an answer for query: {query}")
    response = retrieval_chain.invoke({"input": query})
    return response.get("answer", "No relevant answer found in the context.")

# Function for tool invocation based on the query
def tool_invocation(query, context):
    prompt = llm_prompt.format(input=query, context=context)
    response = llm.invoke(prompt)
    
    # Extract tool calls from response (simple logic to match keywords)
    if "Extract PDF Content" in response:
        return extract_pdf_content(pdf_file_path)
    elif "Classify Question" in response:
        return classify_question(query)
    elif "Create Vector Store" in response:
        return create_vector_db_from_text(context)
    elif "Answer from Query" in response:
        return get_answer_from_query(query, context)
    else:
        return "No relevant tools were invoked."

# Main function to run the process
if __name__ == "__main__":
    # Specify the PDF file path
    pdf_file_path = r"C:\\Users\\hp\\Desktop\\New folder\\paper01.pdf"
    
    print(f"Checking file path: {pdf_file_path}")
    if not os.path.isfile(pdf_file_path):
        print(f"Error: PDF file not found at path {pdf_file_path}")
    else:
        print(f"PDF file found at: {pdf_file_path}")
        try:
            # Use the tools to extract content and create the vector store
            text_documents = extract_pdf_content(pdf_file_path)
            vector_store = create_vector_db_from_text(text_documents)
            
            print("The PDF has been processed. You can now ask questions.")
            while True:
                query = input("Enter your question (or type 'exit' to quit): ")
                if query.lower() == "exit":
                    print("Exiting the program. Goodbye!")
                    break

                # Use the tools to classify and get the answer
                answer = get_answer_from_query(query, vector_store)
                print(f"Answer: {answer}")
        except Exception as e:
            print(f"An error occurred: {e}")
