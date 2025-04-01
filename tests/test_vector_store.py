import pytest
from langchain.schema import Document

from src.ai_core.embeddings import EmbeddingsFactory
from src.ai_core.vector_store import VectorStoreFactory
from src.utils.config_mngr import global_config

global_config().select_config("pytest")


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="The quick brown fox jumps over the lazy dog"),
        Document(page_content="Python is a powerful programming language"),
        Document(page_content="Machine learning is transforming many industries"),
    ]


@pytest.mark.parametrize("vector_store_type", ["InMemory", "Chroma_in_memory"])
def test_vector_store_creation_and_search(sample_documents, vector_store_type) -> None:
    """Test vector store creation, document addition, and similarity search.

    Args:
        sample_documents: Fixture providing test documents
        vector_store_type: Parametrized vector store type to test
    """
    # Create vector store factory
    vs_factory = VectorStoreFactory(
        id=vector_store_type,
        embeddings_factory=EmbeddingsFactory(),
    )

    # Add documents
    db = vs_factory.vector_store
    db.add_documents(sample_documents)

    # Perform similarity search
    query = "programming language"
    results = db.similarity_search(query, k=2)

    assert len(results) == 2
    assert any("Python" in doc.page_content for doc in results)


def test_vector_store_factory_methods() -> None:
    """Test VectorStoreFactory class methods."""
    # Test known items method
    known_stores = VectorStoreFactory.known_items()
    assert isinstance(known_stores, list)
    assert len(known_stores) > 0


def test_vector_store_retriever() -> None:
    """Test vector store retriever functionality."""
    vs_factory = VectorStoreFactory(
        embeddings_factory=EmbeddingsFactory(),
    )
    db = vs_factory.vector_store
    db.add_documents(
        [
            Document(page_content="AI is revolutionizing technology"),
            Document(page_content="Machine learning algorithms are complex"),
        ]
    )

    # Test default retriever
    retriever = vs_factory.as_retriever_configurable(top_k=1)
    results = retriever.invoke("AI technology")

    assert len(results) == 1
