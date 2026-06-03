import pickle, pandas as pd

train = pickle.load(open('data/train_interactions.pkl','rb'))
test = pickle.load(open('data/test_interactions.pkl','rb'))
embeddings = pickle.load(open('embeddings/item_embeddings.pkl','rb'))
long_tail = pickle.load(open('data/long_tail_ids.pkl','rb'))

# 1. Basic shapes
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Unique users in train:", train['user_id'].nunique())
print("Unique track_ids in train:", train['track_id'].nunique())
print("Unique track_ids in test:", test['track_id'].nunique())
print("Items in embedding map:", len(embeddings))
print("Long-tail items count:", len(long_tail))

# 2. Overlap check — THIS IS THE KEY DIAGNOSTIC
train_tracks = set(train['track_id'].unique())
test_tracks = set(test['track_id'].unique())
emb_tracks = set(embeddings.keys())

print("\nTest tracks that appear in train:",
      len(test_tracks & train_tracks))
print("Test tracks that appear in embeddings:",
      len(test_tracks & emb_tracks))
print("Train tracks that appear in embeddings:",
      len(train_tracks & emb_tracks))

# 3. Sample one user end-to-end
sample_user = train['user_id'].iloc[0]
user_train = train[train.user_id == sample_user].track_id.tolist()
user_test = test[test.user_id == sample_user].track_id.tolist()

print(f"\nSample user: {sample_user}")
print(f"Train items: {len(user_train)}")
print(f"Test items: {len(user_test)}")
print(f"Test items in train: {len(set(user_test) & set(user_train))}")
print(f"Test items in embeddings: {len(set(user_test) & emb_tracks)}")
print(f"Sample test track_ids: {user_test[:5]}")
print(f"Sample embedding keys: {list(emb_tracks)[:5]}")

# 4. Long-tail overlap with test
lt_set = set(long_tail)
print(f"\nLong-tail items in test set: {len(test_tracks & lt_set)}")
print(f"Long-tail threshold (max plays): check track_popularity")

# Extra diagnostics
print("\n--- EXTRA TYPE DIAGNOSTICS ---")
print(f"train track_id dtype: {train['track_id'].dtype}")
print(f"test track_id dtype:  {test['track_id'].dtype}")
sample_emb_key = list(emb_tracks)[:1]
print(f"embedding key type:   {type(sample_emb_key[0]) if sample_emb_key else 'N/A'}")
print(f"sample embedding key: {sample_emb_key}")
print(f"sample train track_id: {list(train_tracks)[:1]}")
print(f"sample test track_id:  {list(test_tracks)[:1]}")
