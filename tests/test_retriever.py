from src.retriever import Retriever


def test_retriever_search():
    documents = [
        "RAG combines retrieval with language generation.",
        "Embeddings represent text as numerical vectors.",
        "Dublin is the capital of Ireland.",
    ]

    retriever = Retriever(documents)

    results = retriever.search(
        "How does RAG work?",
        k=2,
    )

    assert len(results) == 2
    assert results is not None