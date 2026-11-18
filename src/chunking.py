
import re


def chunk_text(text, chunk_size=200, overlap=1):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    chunks = []
    current = []
    current_length = 0

    for sentence in sentences:
        new_length = current_length + len(sentence)
        if current:
            new_length += 1

        if current and new_length > chunk_size:
            chunks.append(" ".join(current))

            # Keep overlapping sentences
            current = current[-overlap:] if overlap else []
            current_length = len(" ".join(current))

            # Avoid an overlap that leaves no room
            while current and current_length + len(sentence) + 1 > chunk_size:
                current.pop(0)
                current_length = len(" ".join(current))

        current.append(sentence)
        current_length = len(" ".join(current))

    if current:
        chunks.append(" ".join(current))

    return chunks


if __name__ == "__main__":
    text = (
        "Retrieval-Augmented Generation combines "
        "information retrieval and language generation. "
        "A retriever finds relevant passages from a "
        "knowledge base. The language model then uses "
        "those passages to generate an answer."
    )

    for i, chunk in enumerate(chunk_text(text, chunk_size=100)):
        print(f"Chunk {i + 1}: {chunk}\n")
