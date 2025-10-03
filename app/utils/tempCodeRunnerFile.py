
def process_audio_pipeline(input_audio: str):
    print("\nRunning STT...")
    query_text = transcribe_audio(input_audio)
    # query_text = "What benefits will I get in the education sector as a blind individual?"
    print("User Query:", query_text)

    print("\nRunning RAG pipeline...")
    answer_text, retrieved_chunks = rag_pipeline(query_text)
    print("\nRAG Answer:", answer_text)

    print("\nGenerating speech...")