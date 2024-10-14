import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the LLM model and tokenizer
llm_model = T5ForConditionalGeneration.from_pretrained('t5-base')
llm_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Define the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Chatbot")
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

# Text extraction from documents
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    return text

# Text chunking
def chunk_text(text, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Get the best matching chunk
def get_best_match(query, chunks, model):
    query_embedding = model.encode(query)
    chunk_embeddings = model.encode(chunks)
    similarities = cosine_similarity([query_embedding], chunk_embeddings)
    best_match_index = similarities.argmax()
    return chunks[best_match_index]

# Generate response using LLM
def get_llm_response(query, context):
    input_text = f"question: {query} context: {context}"
    inputs = llm_tokenizer(input_text, return_tensors="pt")
    outputs = llm_model.generate(**inputs)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Integrate chatbot into Streamlit
if uploaded_file is not None:
    text = extract_text(uploaded_file)
    chunks = chunk_text(text)
    st.write("Document processed successfully!")

    query = st.text_input("Ask a question about the document:")
    if query:
        best_match = get_best_match(query, chunks, embedding_model)
        llm_response = get_llm_response(query, best_match)
        st.write("Best Match:", best_match)
        st.write("LLM Response:", llm_response)
