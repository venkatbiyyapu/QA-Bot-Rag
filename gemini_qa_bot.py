import time
import os
import pinecone
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pinecone import ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI using the API key from environment
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone client using the API key from environment
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get the Pinecone index name from environment variables
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize the Google Generative AI Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Check if Pinecone index exists; if not, create a new one
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,  # Set the embedding dimension to 768
        metric="cosine",  # Use cosine similarity metric
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'  # Specify the cloud and region
        ) 
    ) 

# Function to extract text from uploaded PDF files
def get_pdf_text_extractor(pdf_docs):
    text = ""
    # Loop through each uploaded PDF document
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Initialize PDF reader
        # Extract text from each page of the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split extracted text into smaller chunks
def get_chunks(text):
    # Use RecursiveCharacterTextSplitter to split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    return chunks

# Function to add text chunks to Pinecone vector database
def add_to_pinecone_db(text_chunks):
    # Use Pinecone vector store to add chunks
    PineconeVectorStore.from_texts(index_name=index_name, texts=text_chunks, embedding=embeddings, text_key='text')

# Function to get the conversational QA chain
def get_conversational_chain():
    # Define a prompt template for the conversational model
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    # Initialize the Gemini Pro model from Google Generative AI
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Create a prompt with the input variables 'context' and 'question'
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the QA chain with the 'stuff' method (used when all documents are concatenated)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and process the question
def user_input(user_question):
    # Initialize Pinecone vector store and search for the top 5 relevant documents
    new_docs = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings, text_key='text')
    docs = new_docs.similarity_search(user_question, k=5)  # Perform similarity search
    print(len(docs))  # Print the number of documents found

    # Get the conversational chain
    chain = get_conversational_chain()
    # Run the chain with the documents and question
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True
    )

    print(response)  # Print the response
    return response["output_text"]

# Function to generate a response stream (yields one word at a time)
def response_generator(res):
    for word in res.split(" "):
        yield word + " "  # Yield word with a space
        time.sleep(0.04)  # Delay for streaming effect

# Main function to set up the Streamlit app
def main():
    res = ''
    # Set the page configuration for Streamlit
    st.set_page_config("Chat PDF")
    st.header("Upload Pdf's and Chat with QA Bot")  # Set header

    # Initialize message state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input and process it
    if user_question := st.chat_input("Ask a Question from the PDF Files"):
        with st.chat_message("user"):
            st.markdown(user_question)  # Display user input in chat
        st.session_state.messages.append({"role": "user", "content": user_question})
        res = user_input(user_question)  # Get the response
    
        # Stream the assistant's response
        with st.chat_message("assistant"):
            st.write_stream(response_generator(res))  # Write response in streaming format

        # Add the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": res})

    # Sidebar menu for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        # File uploader for PDFs
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text_extractor(pdf_docs)  # Extract text from uploaded PDFs
                text_chunks = get_chunks(raw_text)  # Split the text into chunks
                add_to_pinecone_db(text_chunks)  # Add the chunks to Pinecone database
                st.success("Done")  # Display success message

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
