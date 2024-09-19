import warnings
warnings.filterwarnings('ignore')

import os
import gradio as gr

import torch
import tempfile
import numpy as np
import cohere
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from nltk.tokenize.texttiling import TextTilingTokenizer
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Download spaCy model
spacy.cli.download("en_core_web_sm")

co = cohere.Client(os.environ.get("CO_API_KEY"))

nlp = spacy.load("en_core_web_sm")

from transformers import AutoTokenizer, AutoModel

# Load models
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

# Initialize TextTilingTokenizer with default parameters
tiling_tokenizer = TextTilingTokenizer()

def generate_response(prompt, embeddings):
    aggregated_embedding = np.mean([np.mean(embed) for embed in embeddings])
    embedding_str = f"Embedding summary: {aggregated_embedding:.2f}"
    
    full_prompt = f"{embedding_str}\n\n{prompt}"
    
    try:
        response = co.generate(
            model="command-xlarge-nightly",
            prompt=full_prompt,
            max_tokens=750  # Increase the max tokens for a longer response
        )
        return response.generations[0].text.strip()
    
    except cohere.error.CohereError as e:
        return f"An error occurred: {str(e)}"

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def get_bert_embeddings(texts):
    embeddings_list = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings_list.append(embeddings)

    return embeddings_list

def analyze_text(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    return entities, pos_tags, dependencies

def process_pdf_and_generate_response(pdf_file, query):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        with open(pdf_file, 'rb') as f:
            temp_file.write(f.read())
            temp_file_path = temp_file.name

    document_text = extract_text_from_pdf(temp_file_path)

    entities, pos_tags, dependencies = analyze_text(document_text)
    
    print("Entities:", entities)
    print("POS Tags:", pos_tags)
    print("Dependencies:", dependencies)
    
    # Segment the document text using TextTiling
    text_chunks = tiling_tokenizer.tokenize(document_text)
    
    # Process document text with InLegalBERT
    document_embeddings = get_bert_embeddings(text_chunks)
    
    # Construct prompt for LLM
    prompt = f"You are an AI driven research engine for commercial courts, Given the legal document: '{document_text[:2000]}', answer the query : '{query}'"
    
    # Generate response using LLM
    response = generate_response(prompt, document_embeddings)
    
    return response

def chunk_long_sentence(sentence):
    words = sentence.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(' '.join(current_chunk + [word])) <= 512:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
    
iface = gr.Interface(
    fn=process_pdf_and_generate_response,
    inputs=[
        gr.File(label="Upload PDF Document"), 
        gr.Textbox(label="Query", placeholder="Enter your query here...")
    ],
    outputs=gr.Textbox(label="Response"),
    title="AI-Driven Research Engine for Commercial Courts",
    description="Upload a PDF document and ask a question to get a response generated based on the content of the document."
)

# Launch the interface

    
    
iface.launch(share=True)
