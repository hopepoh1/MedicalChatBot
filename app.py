import streamlit as st
import asyncio
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_api_env = os.environ.get('PINECONE_API_ENV')
Pinecone(api_key=pinecone_api_key, environment=pinecone_api_env)

# Set up Pinecone index
index_name = "medical-bot"
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Set up LLM
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

# Set up QA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 1}),  # Fetch only one document per query
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Streamlit app starts here
st.title("Langchain Application")

input_text = st.text_input("Input: ")  

async def ask_question(input_text):
    result = qa({"query": input_text})
    return result["result"]

if st.button("Ask the question"):
    result = asyncio.run(ask_question(input_text))
    st.subheader("The Response is")
    st.write(result)
