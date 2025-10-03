
"""PINECONE-SETUP

"""

from pinecone import Pinecone, ServerlessSpec
from collections import Counter
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
import uuid
import json
from typing import List, Dict, Tuple
import glob
import numpy as np

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from deepgram import DeepgramClient, SpeakOptions
import asyncio
import openai
from IPython.display import display
from IPython.display import Markdown

# python -m ensurepip --upgrade
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
genai_api_key = os.getenv("GENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))


model = SentenceTransformer('all-MiniLM-L6-v2')

pc = Pinecone(api_key=pinecone_api_key)
genai.configure(api_key=genai_api_key)

# index_name = "text-embeddings"
index_name = "legal-voice-bot"


# # Print the list of existing indexes for debugging
# print("Existing indexes:", pc.list_indexes())

if any(index_info["name"] == index_name for index_info in pc.list_indexes()):
    print(f"Index '{index_name}' already exists. Connecting to it.")
    # pc.delete_index(index_name)
    # print(f"Deleted old index '{index_name}'")
    index = pc.Index(index_name)
else:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    index = pc.Index(index_name)

nltk.download('punkt_tab')
nltk.download('punkt', quiet=True)

# --- Configurable parameters ---
CHARS_PER_CHUNK = 3000   # target characters per chunk (approx)
CHARS_OVERLAP = 300     # overlap between consecutive chunks
MIN_CHUNK_CHARS = 1500    # minimum chars to allow (avoid tiny fra))

"""SPLIT INTO CHUNKS AND GENERATE EMBEDDINGS - STORE IN DB"""

from unstructured.partition.auto import partition
def table_to_text(table_element) -> str:
    """
    Convert a Table element into a readable string.
    Each row is turned into "header: value" pairs joined by semicolons.
    """
    lines = table_element.text.splitlines()
    if not lines:
        return ""
    # Assume first line is header row
    headers = lines[0].split()
    rows_text = []
    for line in lines[1:]:
        cells = line.split()
        pairs = [f"{h}: {v}" for h, v in zip(headers, cells)]
        rows_text.append("; ".join(pairs))
    return "\n".join(rows_text)

def parse_pdf_to_chunks(pdf_path: str, doc_id: str):
    """
    Parse a legal PDF into structured chunks based on headings.
    Returns a list of {'doc_id', 'heading', 'text', 'chunk_id'}.
    """
    # Partition the PDF into elements (Title, NarrativeText, Table, etc.)
    elements = partition(filename=pdf_path, strategy='hi_res')
    chunks = []
    current_heading = None
    current_texts = []
    chunk_counter = 0

    for el in elements:
        el_type = el.type  # e.g., 'Title', 'NarrativeText', 'Table'
        el_text = el.text.strip() if el.text else ""
        if el_type == "Title":
            # Save previous chunk if exists
            if current_heading is not None:
                chunks.append({
                    "doc_id": doc_id,
                    "heading": current_heading,
                    "text": "\n\n".join(current_texts).strip(),
                    "chunk_id": f"{doc_id}_chunk{chunk_counter}"
                })
                chunk_counter += 1
            # Start new chunk with this heading
            current_heading = el_text
            current_texts = []
        elif el_type in ("NarrativeText", "ListItem", "UncategorizedText"):
            # Append narrative or list text under current heading
            if current_heading is None:
                # If no heading seen yet, create a placeholder
                current_heading = ""
            if el_text:
                current_texts.append(el_text)
        elif el_type == "Table":
            # Convert table to string and append
            table_str = table_to_text(el)
            if current_heading is None:
                current_heading = ""
            if table_str:
                current_texts.append(table_str)
        else:
            # Skip other element types (Headers, Footers, PageBreak, etc.)
            continue

    # Add final chunk after loop
    if current_heading is not None and current_texts:
        chunks.append({
            "doc_id": doc_id,
            "heading": current_heading,
            "text": "\n\n".join(current_texts).strip(),
            "chunk_id": f"{doc_id}_chunk{chunk_counter}"
        })

    return chunks

def embed_text(text: str) -> list:
    """
    Generate an embedding vector for the given text using OpenAI.
    """
    if not text:
        return []
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response['data'][0]['embedding']

def upsert_chunks_to_pinecone(chunks: list):
    """
    Embed each chunk and upsert into Pinecone with metadata.
    """
    for chunk in chunks:
        text = chunk["text"]
        vec = embed_text(text)
        metadata = {
            "doc_id": chunk["doc_id"],
            "heading": chunk["heading"],
            "chunk_id": chunk["chunk_id"],
            "text": text
        }
        # Use a unique ID for the vector (e.g., combine doc and chunk)
        vec_id = f"{chunk['doc_id']}_{chunk['chunk_id']}"
        # Upsert vector and metadata into Pinecone
        index.upsert(vectors=[(vec_id, vec, metadata)])

def process_all_pdfs(folder_path="app\\data"):
     for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            doc_id = os.path.splitext(filename)[0]  # e.g., "doc1"
            
            print(f"\n📄 Processing {filename} ...")
            
            # Step 1: Parse into structured chunks
            chunks = parse_pdf_to_chunks(pdf_path, doc_id)
            print(f"   - Extracted {len(chunks)} chunks")
            
            # Step 2: Upsert into Pinecone
            upsert_chunks_to_pinecone(chunks)
            print(f"   - Stored {len(chunks)} chunks in Pinecone ✅")

# print("\n✅ All PDFs from app/data processed and stored in Pinecone.")
# Example usage:
# chunks = parse_pdf_to_chunks("legal_doc.pdf", "doc1")
# upsert_chunks_to_pinecone(chunks)

def expand_query(query: str, n_variations: int = 5) -> list:
    """
    Generate semantically similar query variations using an LLM.
    Returns a list containing the original query plus the variations.
    """
    prompt = (
        f"Generate {n_variations} queries that are semantically similar to the query: \"{query}\".\n"
        "Each query should be a paraphrase or related variation."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates query variations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    text = response.choices[0].message.content.strip()
    # Split into lines and clean formatting
    variants = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    # Ensure the original query is included
    final_queries = [query] + variants[:n_variations]
    return final_queries

def get_final_context(query: str, top_k_per_query: int = 5):
    """
    Expand the query, retrieve chunks from Pinecone, and return the top-3 combined context.
    Returns:
      - final_context: concatenated text of top-3 chunks
      - final_context_chunks: list of the top-3 chunk metadata dicts
    """
    print("\n[Generating Query Variations]")
    print(f"Original Query: {query}")
    # Step a: Query expansion
    similar_queries = expand_query(query)
    for i, sq in enumerate(similar_queries, 1):
        print(f"  {i}. {sq}")
    
    # Step b/c: Embed queries and retrieve top chunks
    all_matches = []
    for q in similar_queries:
        # Embed the query
        q_vec = openai.Embedding.create(model="text-embedding-3-large", input=q)['data'][0]['embedding']
        # Retrieve top chunks from Pinecone
        res = index.query(vector=q_vec, top_k=top_k_per_query, include_metadata=True)
        chunk_ids = [m['metadata']['chunk_id'] for m in res['matches']]
        print(f"\n Query: '{q}'")
        print(f"   Retrieved Chunk IDs: {chunk_ids}")

        for match in res['matches']:
            meta = match['metadata']
            score = match['score']
            # Record each match
            all_matches.append((score, meta))
    
    # Step d: Rank and select top unique chunks by highest score
    # Use a dict to keep best score per chunk_id
    best_chunks = {}
    for score, meta in all_matches:
        key = (meta['doc_id'], meta['chunk_id'])
        # Keep the match with the highest score for each chunk
        if key not in best_chunks or best_chunks[key][0] < score:
            best_chunks[key] = (score, meta)
    # Sort by score descending and take top 3
    top_items = sorted(best_chunks.values(), key=lambda x: x[0], reverse=True)[:3]
    
    # Prepare final output
    final_context_chunks = []
    final_texts = []
    top_chunk_ids=[]
    for score, meta in top_items:
        final_context_chunks.append({
            "doc_id": meta['doc_id'],
            "heading": meta['heading'],
            "chunk_id": meta['chunk_id'],
            "text": meta['text']
        })
        final_texts.append(meta['text'])
        top_chunk_ids.append(meta['chunk_id'])
    final_context = "\n\n".join(final_texts)

    print(f"\n[Top 3 Chunks Selected for Final Context]: {top_chunk_ids}")
    print("\n Final Context:\n", final_context)
    return final_context, final_context_chunks

# Example usage:
# context, chunks = get_final_context("breach of contract remedies", top_k_per_query=5)
# print(context)
# print(chunks)


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,  # Token size per chunk
#     chunk_overlap=30,  # Overlap between chunks
#     separators=["\n\n", "\n"],  # Preferred text splitting points
# )

# # Step 5: Extract text from a PDF file
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page_num in range(doc.page_count):
#             page = doc.load_page(page_num)
#             text += page.get_text("text")
#     return text

# # Step 6: Generate embeddings for a chunk of text
# def generate_embedding(text):
#     return model.encode(text).tolist()  # Convert the numpy array to a list for compatibility with Pinecone

# # Step 7: Process PDFs, generate embeddings, and store them in Pinecone
# def process_and_store_in_pinecone(pdf_path):
#     print(f"Processing {pdf_path}...")

#     # Extract text from the PDF
#     full_text = extract_text_from_pdf(pdf_path)

#     # Split the extracted text into smaller chunks
#     chunks = text_splitter.split_text(full_text)
#     print(f"Split into {len(chunks)} chunks.")
#     print(chunks[0:3])

#     # For each chunk, generate an embedding and store it in Pinecone
#     for i, chunk in enumerate(chunks):
#         embedding = generate_embedding(chunk)

#         # Create a unique ID for the chunk (combination of file name and chunk index)
#         unique_id = f"{os.path.basename(pdf_path)}_{i}"

#         # Store the embedding in Pinecone (with metadata)
#         index.upsert([(unique_id, embedding, {"text": chunk, "document_name": os.path.basename(pdf_path), "chunk_id": i})])

#     print(f"Processed and stored embeddings for {os.path.basename(pdf_path)}")

# # Step 8: Process all PDFs in a specified folder
# input_folder = "./data"  # Replace with the actual path to your PDF folder in Google Colab
# pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

# # Process each PDF file and store its embeddings in Pinecone
# for pdf in pdf_files:
#     process_and_store_in_pinecone(pdf)

# print("All documents have been processed and embeddings are stored in Pinecone!")

# query_text = "what can javascript do?"
# query_embedding = generate_embedding(query_text)

# # Perform a search for the most similar documents in Pinecone
# query_result = index.query(vector=query_embedding, top_k=5, include_metadata=True)

# # Display the search results
# print("Search Results:")
# for match in query_result["matches"]:
#     print(f"Result: {match['id']} with score {match['score']}")

"""
Rpwd 2016 RAG pipeline (GENAI + Pinecone)
- Extract text from PDF (PyMuPDF)
- Logical chunking (heading-aware + sentence-based + overlap)
- Create embeddings using SentenceTransformers
- Store in Pinecone

Usage:
  - Set environment variables: PINECONE_API_KEY, GENAI_API_KEY
  - pip install -r requirements.txt
  - python rpwd2016_rag_pipeline.py /path/to/RPwd-Disabilities-2016.pdf

Requirements:
fitz (PyMuPDF), nltk, sentence-transformers, pinecone-client, google-generativeai, python-dotenv
"""



# --- PDF extraction ---
def extract_text_by_page(pdf_path: str) -> List[str]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            pages.append(text or "")
    return pages

# --- Heading detection heuristics ---
HEADING_RE = re.compile(r"^\s*(Section|CHAPTER|Chapter|ANNEX|ANNEXURE|Article|Schedule|\d+\.|[A-Z][A-Z\s]{4,})")


def detect_headings_in_text(text: str) -> List[Tuple[int, str]]:
    headings = []
    lines = text.splitlines()
    idx = 0
    for ln in lines:
        ln_stripped = ln.strip()
        if not ln_stripped:
            idx += len(ln) + 1
            continue
        if HEADING_RE.match(ln_stripped):
            headings.append((idx, ln_stripped))
        else:
            if len(ln_stripped) <= 120 and sum(1 for ch in ln_stripped if ch.isalpha()) >= 3:
                letters = [ch for ch in ln_stripped if ch.isalpha()]
                if letters and all(ch.isupper() for ch in letters[:min(10, len(letters))]):
                    headings.append((idx, ln_stripped))
        idx += len(ln) + 1
    return headings

# --- Chunking ---
def chunk_by_headings(full_text: str) -> List[Dict]:
    headings = detect_headings_in_text(full_text)
    if not headings:
        return [{
            'id': str(uuid.uuid4()),
            'title': 'full_document',
            'text': full_text,
            'start_char': 0,
            'end_char': len(full_text)
        }]

    sections = []
    for i, (pos, title) in enumerate(headings):
        start = pos
        end = headings[i+1][0] if i+1 < len(headings) else len(full_text)
        sect_text = full_text[start:end].strip()
        sections.append({'title': title, 'text': sect_text, 'start_char': start, 'end_char': end})

    final_chunks = []
    for sec in sections:
        sec_text = sec['text']
        if len(sec_text) <= CHARS_PER_CHUNK:
            final_chunks.append({
                'id': str(uuid.uuid4()),
                'title': sec['title'],
                'text': sec_text,
                'start_char': sec['start_char'],
                'end_char': sec['end_char']
            })
            continue

        sentences = sent_tokenize(sec_text)
        cur = ""
        cur_start = None
        char_cursor = 0
        for sent in sentences:
            if cur == "":
                cur_start = sec_text.find(sent, char_cursor)
            cur += (" " if cur else "") + sent
            char_cursor = sec_text.find(sent, char_cursor) + len(sent)

            if len(cur) >= CHARS_PER_CHUNK:
                if len(cur) < MIN_CHUNK_CHARS and final_chunks:
                    final_chunks[-1]['text'] += "\n" + cur
                    final_chunks[-1]['end_char'] = sec['start_char'] + char_cursor
                else:
                    final_chunks.append({
                        'id': str(uuid.uuid4()),
                        'title': sec['title'],
                        'text': cur,
                        'start_char': sec['start_char'] + (cur_start if cur_start is not None else 0),
                        'end_char': sec['start_char'] + char_cursor
                    })
                overlap_text = cur[-CHARS_OVERLAP:]
                cur = overlap_text
                cur_start = sec_text.find(overlap_text, char_cursor - len(overlap_text))

        if cur.strip():
            if len(cur) < MIN_CHUNK_CHARS and final_chunks:
                final_chunks[-1]['text'] += "\n" + cur
                final_chunks[-1]['end_char'] = sec['end_char']
            else:
                final_chunks.append({
                    'id': str(uuid.uuid4()),
                    'title': sec['title'],
                    'text': cur,
                    'start_char': sec['end_char'] - len(cur),
                    'end_char': sec['end_char']
                })

    return final_chunks

# --- Embedding + Pinecone upsert ---
def embed_and_upsert(chunks: List[Dict], doc_id: str, document_name:str="RPwD_2016"):
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts).tolist()

    vectors = []
    for c, emb in zip(chunks, embeddings):
        meta = {
            'document_name': document_name,
            'doc_id': doc_id,
            'title': c.get('title'),
            'start_char': c.get('start_char'),
            'end_char': c.get('end_char'),
            'preview': c['text'][:320],
        }
        vectors.append({
            "id": c['id'],
            "values": emb,
            "metadata": meta
        })

    index.upsert(vectors)
    print(f"Upserted {len(vectors)} chunks into Pinecone index '{index_name}'")

    manifest_path = f"{doc_id}_chunks_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Wrote chunk manifest to {manifest_path}")

# --- Orchestration ---
import os

def process_all_pdfs_in_folder(folder_path: str = "app\\data"):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            doc_id = os.path.splitext(filename)[0]

            print(f" Processing {filename} ...")

            try:
                pages = extract_text_by_page(pdf_path)
                full_text = "\n\n".join(pages)

                chunks = chunk_by_headings(full_text)
                embed_and_upsert(chunks, doc_id, document_name=doc_id)

                print(f" Successfully processed {filename}")
            except Exception as e:
                print(f" Error processing {filename}: {e}")

    


"""
CREATING SIMILAR QUERIES-BUILDING THE CONTEXT
"""

def generate_similar_queries(query, num_similar_queries=5):
    # Augment the query with the refined prompt
    prompt = f"""
    Given the following query, generate {num_similar_queries} similar queries that:

    1. Use **synonyms** or related terms.
    2. Maintain the **original context** and meaning of the query.
    3. Consider alternative **phrasing** and **wording** that would likely retrieve the same or similar documents.
    4. Include different aspects: abstraction, specificity, technical details, temporal context, comparisons.
    5. **Do not include any empty responses** or irrelevant text. Each query should be a complete and meaningful sentence.
    6. Ensure that the generated queries are distinct from one another, each offering a new variation on the original query.
    7. The queries should be such that it will be used by a RAG system

    Query: "{query}"

    Generate {num_similar_queries} alternative queries, with each query separated by a new line:
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    augmented_queries = [q.strip() for q in response.text.split("\n") if q.strip()]

    all_queries = [query] + augmented_queries
    print("\n[Similar Queries Generated]:")
    for i, q in enumerate(all_queries):
        print(f"Query {i+1}: {q}")
    return all_queries


def embed_text(text):
    return model.encode(text).tolist()


def search_similar_documents_pinecone(query, top_k=5):
    similar_queries = generate_similar_queries(query)
    all_results = []
    per_query_chunks = []

    for idx, q in enumerate(similar_queries):
        embedding = embed_text(q)
        if not embedding:
            print("Error generating query embedding.")
            continue

        search_results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )

        results = []
        for match in search_results["matches"]:
            metadata = match.get("metadata", {})
            result = {
                "id": match.get("id"),
               "score": match.get("score"),
                "doc_id": metadata.get("doc_id", "N/A"),
                "document_name": metadata.get("document_name", "N/A"),
                "title": metadata.get("title", ""),
                "preview": metadata.get("preview", ""),
                "start_char": metadata.get("start_char"),
                "end_char": metadata.get("end_char")
            }
            results.append(result)

        all_results.extend(results)
        per_query_chunks.append(results)

        # Print top chunks
        print(f"\n[Top {top_k} Chunks for Query {idx+1}]: {q}")
        # for i, chunk in enumerate(results):
        #     print(f"  Chunk {i+1}: ID={chunk['id']}, Doc={chunk['document_name']}, Score={chunk['score']}")
        for i, chunk in enumerate(search_results['matches']):
            print(f"  Chunk {i+1}: ID={chunk['id']}, Doc={chunk['metadata']['document_name']}, Score={chunk['score']}")


    # Count references across all queries
    document_counts = Counter(doc["document_name"] for doc in all_results)
    print("\n[Document References Across All Queries]:")
    for doc, count in document_counts.items():
        print(f"{doc}: {count} times")

    return all_results


def get_final_context(query):
    """Full RAG process: generate similar queries, search Pinecone, and return results."""
    results = search_similar_documents_pinecone(query)

    # Step 1: Count frequency of each chunk_id
    chunk_freq = Counter([doc["id"] for doc in results])

    # Step 2: Get the 3 most common chunk_ids
    top_chunks = [id for id, _ in chunk_freq.most_common(3)]
    print(f"\n[Top 3 Most Common Chunk IDs Across All Queries]: {top_chunks}")

    # Step 3: Extract the text of those chunks
    final_context_chunks = []
    seen = set()
    for doc in results:
        if doc["id"] in top_chunks and doc["id"] not in seen:
            final_context_chunks.append(doc)  
            seen.add(doc["id"])

    print(f"\n[Final Context Chunks Used]:")
    for i, chunk in enumerate(final_context_chunks):
        preview = doc.get("preview", doc.get("text", ""))
        print(f"Chunk {i+1}: {preview[:100]}{'...' if len(preview) > 100 else ''}")

    # Step 4: Join chunks into context string
    final_context = "\n\n".join([d.get("preview", d.get("text", "")) for d in final_context_chunks])
    return final_context, final_context_chunks


def get_answer_from_llm(query, context):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer the question using the given context. Exclude any asterisk or special characters except words. Include information from external sources only if the context is insufficient, you can refer to external knowledge or trusted internet sources to find your answer.
    Don't say "according to the context" or "based on the context" in your answer. Answer like you are the legal advisor. Give proper explanation for your answer and use good language.
i want maximum characters to be 1900.

    """
    response = model.generate_content(prompt)
    if hasattr(response, "candidates") and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    elif hasattr(response, "text"):
        return response.text.strip()
    else:
        return str(response).strip()
    


def rag_pipeline(query: str) -> str:
    # Replace these with your RAG functions
    context, retrieved_chunks = get_final_context(query)
    answer = get_answer_from_llm(query, context)
    return answer, retrieved_chunks


# === Step 1: Speech-to-Text ===
def transcribe_audio(file_path: str) -> str:
    audio_file = genai.upload_file(path=file_path, display_name="user audio")
    uploaded_audio_file = genai.get_file(name=audio_file.name)

    # Use a model that supports audio input
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Prompt to transcribe and translate to English
    prompt = "Translate the content of this audio into English."

    try:
        # Send the prompt and the audio file to the model
        response = model.generate_content([prompt, uploaded_audio_file])

        # Display and return the translated text
        display(Markdown(response.text))
        return response.text

    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


# === Step 2: Text-to-Speech ===
from deepgram import DeepgramClient, SpeakOptions

def synthesize_speech(text: str, filename: str = "output.wav") -> str:
    """
    Converts text to speech using Deepgram TTS and saves it as a WAV file.
    """
    t = {"text": text}
    FILENAME = "audio.mp3"

    try:
        options = SpeakOptions(
            model="aura-2-thalia-en",
        )

        response = deepgram.speak.v("1").save(
            FILENAME,
            t,
            options,
        )

        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")




def evaluate_response(query, answer_text, retrieved_chunks):
    # Retrieval confidence
    retrieval_confidence = np.mean([c["score"] for c in retrieved_chunks]) if retrieved_chunks else 0

    # Alignment score
    answer_embedding = model.encode(answer_text)
    context_text = " ".join([c.get("preview", "") for c in retrieved_chunks])
    context_embedding = model.encode(context_text)
    alignment_score = float(np.dot(answer_embedding, context_embedding) /
                            (np.linalg.norm(answer_embedding) * np.linalg.norm(context_embedding)))

    # Final score (simple average)
    final_score = (retrieval_confidence + alignment_score) / 2

    return {
        "retrieval_confidence": retrieval_confidence,
        "alignment_score": alignment_score,
        "final_score": final_score
    }

# === Main Pipeline ===
def process_audio_pipeline(input_audio: str):
    print("\nRunning STT...")
    query_text = transcribe_audio(input_audio)
    # query_text = "What benefits will I get in the education sector as a blind individual?"
    print("User Query:", query_text)

    print("\nRunning RAG pipeline...")
    answer_text, retrieved_chunks = rag_pipeline(query_text)
    print("\nRAG Answer:", answer_text)

    print("\nGenerating speech...")
    audio_output = synthesize_speech(answer_text)

    print("\nEvaluating response quality...")
    metrics = evaluate_response(query_text, answer_text, retrieved_chunks)

    print("\n--- Evaluation Metrics ---")
    print(f"Retrieval Confidence: {metrics['retrieval_confidence']:.3f}")
    print(f"Answer–Context Alignment: {metrics['alignment_score']:.3f}")
    print(f"Response Reliability Score: {metrics['final_score']:.3f}")

    print(f"\nPipeline complete. Answer saved as {audio_output}")


if __name__ == "__main__":
    # Replace with your test audio file path
    input_file = "app\\data\\input\\army_input.mp3"
    process_audio_pipeline(input_file)
    # process_all_pdfs_in_folder("app\\data")
    # synthesize_speech("Your lab results show  and scheduling a follow-up in eight weeks to reassess.")



