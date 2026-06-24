
print("DEBUG: Script started...")
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from sentinel.agentic_audit import CanaryAuditor, setup_audit_vector_db, embedder
import numpy as np

def test_retrieval_accuracy():
    print("🚀 Initializing RAG Setup Test...")
    
    # 1. Initialize
    auditor = CanaryAuditor()
    
    # 2. Setup Vector DB
    index, docs = setup_audit_vector_db(auditor)
    
    # 3. Define a Search Query
    query = "When can I eat lunch?"
    query_vec = embedder.encode([query])
    
    # 4. Perform Search
    distances, indices = index.search(np.array(query_vec).astype('float32'), k=1)
    retrieved_doc = docs[indices[0][0]]
    
    print(f"\nQuery: {query}")
    print(f"Retrieved: {retrieved_doc}")
    
    if "cafeteria" in retrieved_doc.lower():
        print("\n✅ SUCCESS: Vector Retrieval is working correctly.")
    else:
        print("\n❌ FAILURE: Vector Retrieval returned the wrong document.")

if __name__ == "__main__":
    test_retrieval_accuracy()