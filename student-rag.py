import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DOCS_FOLDER = "doc"
COLLECTION_NAME = "my_rag_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 140
TOP_K = 4
SCORE_THRESHOLD = 0.6
TEMPERATURE = 0.2


def initialize_system():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=6333
    )

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    test_vector = embeddings.embed_query("test")
    vector_dim = len(test_vector)

    try:
        client.get_collection(COLLECTION_NAME)
    except:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    if client.get_collection(COLLECTION_NAME).points_count == 0:
        loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        splits = splitter.split_documents(docs)
        vector_store.add_documents(splits)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": SCORE_THRESHOLD,
            "k": TOP_K,
        },
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=TEMPERATURE,
        num_ctx=4096,
        max_tokens=500,
    )

    prompt_template = """
You are a senior academic expert in Cryptography and Network Security.

Answer ONLY using the provided context from:
"Cryptography and Network Security: Principles and Practice" by William Stallings (4th Edition).

Rules:
- No outside knowledge.
- No hallucination.
- If insufficient context, respond exactly with:
"The provided document sections do not contain enough information to fully answer this."

Context:
{context}

Question:
{question}

Comprehensive Answer:
"""

    prompt = PromptTemplate.from_template(prompt_template)

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(
                f"Source: {d.metadata.get('source','unknown')} | Page: {d.metadata.get('page','?')}\n{d.page_content}"
                for d in docs
            )),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    print("Initializing RAG system...")
    chain = initialize_system()
    print("System Ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask Question: ")
        if query.lower() == "exit":
            break
        answer = chain.invoke(query)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
