
"""PINECONE-SETUP

"""

from pinecone import Pinecone, ServerlessSpec
from collections import Counter
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
import os
import glob
import numpy as np
import fitz

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
genai_api_key = os.getenv("GENAI_API_KEY")

model = SentenceTransformer('all-MiniLM-L6-v2')

pc = Pinecone(api_key=pinecone_api_key)
genai.configure(api_key=genai_api_key)

index_name = "text-embeddings"

# # Print the list of existing indexes for debugging
# print("Existing indexes:", pc.list_indexes())

if any(index_info["name"] == index_name for index_info in pc.list_indexes()):
    # print(f"Index '{index_name}' already exists. Connecting to it.")
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

"""SPLIT INTO CHUNKS AND GENERATE EMBEDDINGS - STORE IN DB"""

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

    # Send the prompt to the model and get the response
    response = model.generate_content(prompt)

    # Split the response by lines and filter out any empty lines
    augmented_queries = [q.strip() for q in response.text.split("\n") if q.strip()]

    # Ensure that the returned list contains the original query as well
    return [query] + augmented_queries

def embed_text(text):
    return model.encode(text).tolist()

def search_similar_documents_pinecone(query, top_k=3):
    similar_queries = generate_similar_queries(query)
    # print(similar_queries)
    all_results = []

    for q in similar_queries:
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
        for match in search_results['matches']:
            result = {
                "score": match["score"],  # Similarity score
                "document_name": match["metadata"]["document_name"],
                "chunk_id": match["metadata"]["chunk_id"],
                "text": match["metadata"]["text"]  # Text chunk for reference
            }
            results.append(result)

        all_results.extend(results)

    document_counts = Counter(doc["document_name"] for doc in all_results)
    print("Document References:")
    for doc, count in document_counts.items():
        print(f"{doc}: {count} times")

    return all_results

def get_final_context(query):
    """Full RAG process: generate similar queries, search Pinecone, and return results."""
    results = search_similar_documents_pinecone(query)
    # Step 1: Count frequency of each chunk_id
    chunk_freq = Counter([doc['chunk_id'] for doc in results])

    # Step 2: Get the 3 most common chunk_ids
    top_chunks = [chunk_id for chunk_id, _ in chunk_freq.most_common(3)]

    # Step 3: Extract the text of those chunks (deduplicated)
    final_context_chunks = []
    seen = set()
    for doc in results:
        if doc['chunk_id'] in top_chunks and doc['chunk_id'] not in seen:
            final_context_chunks.append(doc['text'])
            seen.add(doc['chunk_id'])

    # Step 4: Join them into one final context string
    final_context = "\n\n".join(final_context_chunks)
    # print(seen)

    # print(final_context)
    return final_context

def get_answer_from_llm(query, context):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer the question using only the given context. If the context is insufficient, say you don't have enough information.

    """
    response = model.generate_content(prompt)
    return response.text



