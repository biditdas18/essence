import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded")



def embed_text(text: str) -> np.ndarray:
    """
    Embed text using a pre-trained model.

    Args:
        text (str): The text to embed.

    Returns:
        np.ndarray: The embedded text as a numpy array.
    """
    # Placeholder for actual embedding logic
    # In practice, you would load a model and use it to generate embeddings
    embedding = model.encode(text, normalize_embeddings=True)
    
    return np.array(embedding, dtype=np.float32)