import json
import faiss
import numpy as np
from utils.embedding import embed_text


# Welcome banner
print("\n🎧 Essence – Emotion-Based Music Recommender")
print("Type your feelings and get music that matches your inner world.")
print("Type 'quit' to exit.\n")


# Load FAISS index and song metadata with error handling
try:
    index = faiss.read_index("embeddings/songs.index")
    with open("embeddings/song_metadata.json") as f:
        songs = json.load(f)
except Exception as e:
    print(f"❌ Error loading embeddings or metadata: {e}")
    exit(1)


# Start emotion input loop
k = 3  # Number of nearest neighbors to retrieve
while True:
    user_input = input("Describe your current emotion (or 'quit'): ")

    # Handle empty input
    if not user_input.strip():
        print("⚠️  Please describe an emotion.\n")
        continue

    # Exit condition
    if user_input.lower() == 'quit':
        break

    # Embed user input and reshape for FAISS
    user_embedding = embed_text(user_input)
    D, I = index.search(user_embedding.reshape(1, -1), k)

    # Display results
    print(f"\nTop {k} emotional matches:\n")
    for rank, idx in enumerate(I[0], start=1):
        song = songs[idx]
        print(f"{rank}. 🎵 {song['title']} by {song['artist']}")
        print(f"    {song['emotion_summary']}\n")

# This code snippet is a simple command-line interface that allows users to input their current emotion
# and retrieves the top 3 most similar songs from a pre-built FAISS index. The user input is embedded using
# a pre-trained model, and the FAISS index is used to find the nearest neighbors. The results are printed
# to the console, including the song title, artist, and similarity score. The loop continues until the user
# types 'quit'.
