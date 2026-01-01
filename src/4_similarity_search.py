# src/4_similarity_search.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm

print("=" * 60)
print("🔍 LOTR DIALOG SIMILARITY SEARCH")
print("=" * 60)

# Load data
df = pd.read_csv('../outputs/lotr_with_sentiment.csv')
print(f"📚 Loaded {len(df)} dialog lines")

# Initialize embedding model
print("\n🔄 Loading embedding model (this may take a minute)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

# Generate embeddings
print("\n🔄 Generating embeddings for all dialogs...")
print("⏱️  This will take 3-5 minutes depending on your computer...")

dialogs = df['dialog'].tolist()
embeddings = model.encode(dialogs, show_progress_bar=True, batch_size=32)

print(f"✅ Generated {len(embeddings)} embeddings!")

# Save embeddings for later use
np.save('../outputs/dialog_embeddings.npy', embeddings)
print("✅ Saved: dialog_embeddings.npy")

# Create similarity search function
def find_similar_dialogs(query, top_k=10):
    """Find dialogs similar to the query"""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'character': df.iloc[idx]['char'],
            'dialog': df.iloc[idx]['dialog'],
            'movie': df.iloc[idx]['movie'],
            'sentiment': df.iloc[idx]['sentiment'],
            'similarity': float(similarities[idx])
        })
    
    return results

# Test with famous quotes
print("\n🧪 TESTING SIMILARITY SEARCH")
print("=" * 60)

test_queries = [
    "You shall not pass!",
    "My precious",
    "One does not simply walk into Mordor",
    "I am no man",
    "For Frodo"
]

search_results = {}

for query in test_queries:
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 60)
    
    results = find_similar_dialogs(query, top_k=5)
    search_results[query] = results
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['character']} ({result['similarity']:.3f}):")
        print(f"   {result['dialog'][:80]}...")
        print(f"   Movie: {result['movie']} | Sentiment: {result['sentiment']}")

# Find most similar dialog pairs in the entire dataset
print("\n\n🔗 FINDING MOST SIMILAR DIALOG PAIRS")
print("=" * 60)
print("⏱️  Computing pairwise similarities (this takes a moment)...")

# Sample for efficiency (check 1000 random dialogs)
sample_size = min(1000, len(df))
sample_indices = np.random.choice(len(df), sample_size, replace=False)
sample_embeddings = embeddings[sample_indices]

similarity_matrix = cosine_similarity(sample_embeddings)

# Find top similar pairs (excluding self-similarity)
np.fill_diagonal(similarity_matrix, -1)

most_similar_pairs = []
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        if similarity_matrix[i, j] > 0.8:  # High similarity threshold
            idx1 = sample_indices[i]
            idx2 = sample_indices[j]
            most_similar_pairs.append({
                'char1': df.iloc[idx1]['char'],
                'dialog1': df.iloc[idx1]['dialog'],
                'char2': df.iloc[idx2]['char'],
                'dialog2': df.iloc[idx2]['dialog'],
                'similarity': float(similarity_matrix[i, j])
            })

# Sort by similarity
most_similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

print(f"\n✅ Found {len(most_similar_pairs)} highly similar dialog pairs!")
print("\nTop 10 Most Similar Dialog Pairs:")
print("-" * 60)

for i, pair in enumerate(most_similar_pairs[:10], 1):
    print(f"\n{i}. Similarity: {pair['similarity']:.3f}")
    print(f"   {pair['char1']}: {pair['dialog1'][:60]}...")
    print(f"   {pair['char2']}: {pair['dialog2'][:60]}...")

# Character similarity analysis
print("\n\n👥 CHARACTER SIMILARITY ANALYSIS")
print("=" * 60)

top_chars = df['char'].value_counts().head(8).index.tolist()
char_vectors = {}

for char in top_chars:
    char_dialogs = df[df['char'] == char]
    char_indices = char_dialogs.index.tolist()
    char_embeddings = embeddings[char_indices]
    # Average embedding for character
    char_vectors[char] = np.mean(char_embeddings, axis=0)

print(f"✅ Computed average vectors for {len(char_vectors)} characters")

# Calculate character similarities
char_similarities = {}
for i, char1 in enumerate(top_chars):
    for char2 in top_chars[i+1:]:
        sim = cosine_similarity([char_vectors[char1]], [char_vectors[char2]])[0][0]
        char_similarities[f"{char1} - {char2}"] = float(sim)

# Sort by similarity
sorted_char_sims = sorted(char_similarities.items(), key=lambda x: x[1], reverse=True)

print("\nMost Similar Character Pairs (by dialog style):")
print("-" * 60)
for pair, sim in sorted_char_sims[:10]:
    print(f"  {pair}: {sim:.3f}")

print("\nLeast Similar Character Pairs (by dialog style):")
print("-" * 60)
for pair, sim in sorted_char_sims[-5:]:
    print(f"  {pair}: {sim:.3f}")

# Analyze specific character relationships
print("\n\n🤝 ANALYZING FELLOWSHIP MEMBERS")
print("=" * 60)
fellowship = ['FRODO', 'SAM', 'GANDALF', 'ARAGORN', 'LEGOLAS', 'GIMLI', 'BOROMIR', 'MERRY', 'PIPPIN']
fellowship_in_data = [char for char in fellowship if char in top_chars]

print(f"Fellowship members found in top characters: {', '.join(fellowship_in_data)}")

if len(fellowship_in_data) >= 2:
    print("\nFellowship Dialog Style Similarities:")
    print("-" * 60)
    fellowship_sims = []
    for i, char1 in enumerate(fellowship_in_data):
        for char2 in fellowship_in_data[i+1:]:
            pair_key = f"{char1} - {char2}"
            if pair_key in char_similarities:
                fellowship_sims.append((pair_key, char_similarities[pair_key]))
    
    fellowship_sims.sort(key=lambda x: x[1], reverse=True)
    for pair, sim in fellowship_sims[:10]:
        print(f"  {pair}: {sim:.3f}")

# Find example similar dialogs between characters
print("\n\n💬 EXAMPLE SIMILAR DIALOGS BETWEEN CHARACTERS")
print("=" * 60)

def find_similar_between_characters(char1, char2, top_k=3):
    """Find similar dialogs between two characters"""
    char1_data = df[df['char'] == char1]
    char2_data = df[df['char'] == char2]
    
    if len(char1_data) == 0 or len(char2_data) == 0:
        return []
    
    char1_indices = char1_data.index.tolist()
    char2_indices = char2_data.index.tolist()
    
    char1_embeddings = embeddings[char1_indices]
    char2_embeddings = embeddings[char2_indices]
    
    similarities = cosine_similarity(char1_embeddings, char2_embeddings)
    
    # Find top k most similar pairs
    flat_indices = similarities.flatten().argsort()[-top_k:][::-1]
    
    results = []
    for flat_idx in flat_indices:
        i = flat_idx // len(char2_indices)
        j = flat_idx % len(char2_indices)
        
        results.append({
            'char1': char1,
            'dialog1': char1_data.iloc[i]['dialog'],
            'char2': char2,
            'dialog2': char2_data.iloc[j]['dialog'],
            'similarity': float(similarities[i, j])
        })
    
    return results

# Example: Find similar dialogs between Frodo and Sam
if 'FRODO' in top_chars and 'SAM' in top_chars:
    print("\nFRODO ↔ SAM Most Similar Dialogs:")
    print("-" * 60)
    frodo_sam_similar = find_similar_between_characters('FRODO', 'SAM', top_k=3)
    for i, pair in enumerate(frodo_sam_similar, 1):
        print(f"\n{i}. Similarity: {pair['similarity']:.3f}")
        print(f"   FRODO: {pair['dialog1'][:70]}...")
        print(f"   SAM: {pair['dialog2'][:70]}...")

# Example: Find similar dialogs between Gandalf and Saruman
if 'GANDALF' in top_chars and 'SARUMAN' in top_chars:
    print("\n\nGANDALF ↔ SARUMAN Most Similar Dialogs:")
    print("-" * 60)
    gandalf_saruman_similar = find_similar_between_characters('GANDALF', 'SARUMAN', top_k=3)
    for i, pair in enumerate(gandalf_saruman_similar, 1):
        print(f"\n{i}. Similarity: {pair['similarity']:.3f}")
        print(f"   GANDALF: {pair['dialog1'][:70]}...")
        print(f"   SARUMAN: {pair['dialog2'][:70]}...")

# Save all results - FIXED VERSION
print("\n\n💾 SAVING RESULTS")
print("=" * 60)

# Clean up the data to ensure it's JSON serializable
def clean_for_json(obj):
    """Convert numpy/pandas types to native Python types"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    else:
        return obj

output_data = {
    'test_searches': clean_for_json(search_results),
    'similar_pairs': clean_for_json(most_similar_pairs[:50]),
    'character_similarities': clean_for_json(dict(sorted_char_sims)),
    'search_stats': {
        'total_dialogs': int(len(df)),
        'embedding_dimension': int(embeddings.shape[1]),
        'characters_analyzed': int(len(char_vectors)),
        'similar_pairs_found': int(len(most_similar_pairs))
    }
}

with open('../outputs/similarity_search_results.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("✅ Saved: similarity_search_results.json")

# Verify
with open('../outputs/similarity_search_results.json', 'r') as f:
    test_load = json.load(f)
    print(f"✅ Verified: JSON is valid")