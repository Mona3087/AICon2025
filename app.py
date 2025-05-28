import re
import chainlit as cl  # importing chainlit for our app 
from dotenv import load_dotenv
import tiktoken
import os


from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import openlit
openlit.init(
   otlp_endpoint="http://127.0.0.1:4318", 
)

import logging
logging.basicConfig(level=logging.DEBUG)

# openlit.init(
#     otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
#     otlp_headers=os.getenv("OTLP_HEADERS"),
    
# )

load_dotenv()
# Define a template for generating chat prompts using a given context and query.
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}
Answer the question based on {context}, if you can't figure just say "I don't know"
"""

# Initialize the OpenAI chat model specifically for version gpt-3.5-turbo.
openai_chat_model = ChatOpenAI(model="gpt-4o")
# Obtain the encoding function for the specified model to handle text inputs.
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
# Initialize a model for embedding text, using a smaller, faster version.
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# Load documents from a PDF using the PyMuPDF library
docs = PyMuPDFLoader('AIConUSA2025-ConcurrentSessionTechWell.pdf').load()

# Function to compute the number of tokens in a given text using the tiktoken library.
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
    return len(tokens)


# Define a text splitter that handles character-based splitting with specific overlap and size settings.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10,
    length_function = tiktoken_len,
)
# Split the loaded documents into chunks using the defined text splitter.
split_chunks = text_splitter.split_documents(docs)

# Create a vector store in memory using the Qdrant library, which stores document embeddings.
qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="AI Con",
)
# Convert the vector store into a retriever that can fetch relevant document chunks.
qdrant_retriever = qdrant_vectorstore.as_retriever()

# Template for generating retrieval-augmented prompts.
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# Chain sequence that handles context retrieval, question posing, and response generation.
retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)


# Decorator to mark function execution at the start of a user chat session, initializing the chatbot.
@cl.on_chat_start  
async def start_chat():
    msg=cl.Message(content="Firing up the AI Con 2025 info bot...")
    await msg.send()
    runnable=retrieval_augmented_qa_chain
    msg.content= "Hello, welcome to AI Con 2025. What info would you like about the concurrent sessions?"
    await msg.update()
    cl.user_session.set("runnable",runnable)

@cl.on_message  
async def main(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    if runnable is None:
        raise ValueError("Runnable is not set in user session.")

    inputs = {"question": message.content}
    cb = cl.AsyncLangchainCallbackHandler()
    result = await runnable.ainvoke(inputs, callbacks=[cb])

    if result is None:
        raise ValueError("Result from runnable is None")

    response = result.get("response")
    if response is None:
        raise ValueError("Response is missing from result")

    # If response is an object with attributes like system_fingerprint, ensure validity
    if hasattr(response, 'system_fingerprint') and response.system_fingerprint is None:
        response.system_fingerprint = ""

    # Convert response to string safely
    response_str = str(response)

    pattern = r"content='(.*?)'"
    pattern_new = r'content="(.*?)"'
    match = re.search(pattern, response_str)
    match_new = re.search(pattern_new, response_str)

    extracted_content = ""
    if match:
        extracted_content = match.group(1)
    elif match_new:
        extracted_content = match_new.group(1)
    else:
        extracted_content = "Could not extract content from response."

    await cl.Message(content=extracted_content).send()
