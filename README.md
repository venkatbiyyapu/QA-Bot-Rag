# QA Bot 🚀

## Project Overview

This project is a Question Answering (QA) bot that allows users to upload multiple PDF files and ask questions based on the content of those files. The bot uses a Retrieval-Augmented Generation (RAG) model built with LangChain, employs Gemini embeddings as the Large Language Model (LLM), and utilizes Pinecone as the vector database for efficient information retrieval. The application is developed using Streamlit for a user-friendly interface. 📄🤖

## Folder Structure

```
qa-bot/
├── gemini_qa_bot.py          # Main application file
├── requirements.txt           # List of Python dependencies
├── .env                       # Environment variables (API keys and other configurations)
├── Dockerfile                 # Dockerfile for building the application image
├── docker-compose.yml         # Docker Compose configuration file
└── README.md                  # This README file
```

## Requirements 🛠️

The project requires the following Python packages, which are listed in the `requirements.txt` file:

- langchain
- google-generativeai
- pinecone-client
- PyPDF2
- streamlit
- python-dotenv

You can install the required packages manually or use the `requirements.txt` file to set up your environment.

### Python Version 🐍

This project is built using Python 3.10.

## Setup and Running the Project 🏗️

### Step 1: Install Docker

Before running the application, ensure that Docker is installed on your system. You can download and install Docker from [here](https://www.docker.com/get-started). 🐳

### Step 2: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/venkatbiyyapu/qa-bot.git
cd qa-bot
```

### Step 3: Set Up Environment Variables 🌱

Create a `.env` file in the root directory of the project (if not already present) and add your API keys and configurations:

```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=qa-bot-index
```

### Step 4: Build the Docker Image 🔧

Run the following command to build the Docker image:

```bash
docker-compose build
```

### Step 5: Run the Application 🚀

After the build is complete, start the application using:

```bash
docker-compose up
```

### Step 6: Upload PDF Files 📤

1. Open your web browser and navigate to `http://localhost:8501`.
2. Upload one or more PDF files using the provided upload interface.
3. Click on the "Submit & Process" button.
4. Once you see the success message, you can start asking questions.

### Step 7: Ask Questions ❓

In the chat interface, type your questions based on the uploaded PDFs. The bot will use the RAG model to retrieve relevant information and generate responses.

## Video Demo 🎥

Here is a demo video showcasing the functionality of the QA bot:


https://github.com/user-attachments/assets/0ad32555-4e67-4240-ab6e-0231466c4b77




## Notes 📝

- Ensure that your environment variables are correctly set before running the application.
- The application relies on external APIs (Gemini and Pinecone) for embeddings and vector storage, so make sure your API keys are valid and have the necessary permissions.
