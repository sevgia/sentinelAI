import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SentinelRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def ingest_documents(self, doc_list):
        """Convert text docs into searchable vectors."""
        self.documents = doc_list
        embeddings = self.embedder.encode(doc_list)
        
        # Initialize FAISS Index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Ingested {len(doc_list)} documents into Vector DB.")

    def retrieve(self, query, k=2):
        """Find the most relevant documents for a query."""
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs

    def agent_response(self, query, retrieved_context):
        """
        Simulates the Agentic LLM synthesis. 
        In a real scenario, this would be a call to GPT-4 or Llama 3.
        """
        context_str = "\n".join(retrieved_context)
        # Simple prompt simulation
        prompt = f"System: Use the context below to answer.\nContext: {context_str}\nUser: {query}"
        
        # For the audit, we simulate the LLM's 'thought process'

        return f"Based on the records: {context_str}"