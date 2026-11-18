
from transformers import pipeline
from retriever import Retriever
from chunking import chunk_text

# 1. Source document
document = """
Retrieval-Augmented Generation (RAG) combines
information retrieval with language generation.

A retriever searches a knowledge base to identify
passages relevant to a user's question.

Embedding models transform text into numerical
vectors. Similarity search identifies relevant
passages based on their vector representations.

The retrieved passages are added to the prompt.
A language model then generates an answer
using the retrieved information.

RAG can reduce hallucinations, but it does not
guarantee factual correctness.
"""

# 2. Split document into chunks
chunks = chunk_text(
    document,
    chunk_size=180,
    overlap=1
)

print("Number of chunks:", len(chunks))

# 3. Build retrieval index
retriever = Retriever(chunks)

# 4. Retrieve relevant chunks
question = "How does RAG use embeddings?"

results = retriever.search(question, k=len(chunks))

print("\n=== Retrieval Ranking ===")
for i, result in enumerate(results, 1):
    print(result["text"], result["score"])

context = "\n\n".join(
    f"[Source {i}] {doc}"
    for i, (doc, score) in enumerate(results, 1)
)

print("\nRetrieved Context:")
print(context)

# 5. Load LLM
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

# 6. Generate grounded answer
messages = [
    {
        "role": "system",
        "content": (
            "Answer using only the provided context. "
            "Cite source numbers in your answer. "
            "If the context is insufficient, "
            "say 'I don't know'."
        )
    },
    {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    }
]

output = generator(
    messages,
    max_new_tokens=150,
    do_sample=False
)

answer = output[0]["generated_text"][-1]["content"]

print("\nLLM Answer:")
print(answer)
