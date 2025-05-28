# Document-based Question Answering System

This project implements a retrieval-augmented question-answering system using OpenAI's models and the Qdrant vector database. It is designed to process and understand large documents, specifically for handling queries related to specific topics extracted from a loaded PDF document.

## Features

- **PDF Document Processing**: Load and process PDF documents to extract text for analysis.
- **Tokenization**: Compute the token length of text using Tiktoken, tailored for the GPT-3.5-turbo model.
- **Text Chunking**: Efficiently split text into manageable chunks with overlap for better context.
- **Vector Storage**: Embed text chunks into vectors and store them in-memory for fast retrieval.
- **Question Answering**: Handle user queries by retrieving relevant text chunks and generating answers using OpenAI's models.

## System Architecture

The system uses the following main components:
- `ChatOpenAI`: For generating responses to queries using OpenAI's GPT-3.5-turbo.
- `OpenAIEmbeddings`: For embedding text using a model optimized for performance.
- `PyMuPDFLoader`: For loading and processing PDF documents.
- `Qdrant`: For storing and retrieving text vector embeddings.

## Requirements

- Python 3.11+
- OpenAI API (requires API key)
- Qdrant
- PyMuPDF
- Tiktoken
- Chainlit
- Langchain

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Mona3087/pdf-rag.git
   cd pdf-rag

2. **Run the Application locally**
   ```bash
   pip install -r requirements.txt
   chainlit run app.py -w
