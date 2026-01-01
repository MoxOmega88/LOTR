# web-app/api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Load data and model once at startup
print("Loading model and data...")
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('../outputs/lotr_with_sentiment.csv')
embeddings = np.load('../outputs/dialog_embeddings.npy')
print(f"✅ Loaded {len(df)} dialogs and embeddings")

@app.route('/api/search', methods=['POST'])
def search_dialogs():
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Generate query embedding
        query_embedding = model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top results
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
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'total_dialogs': len(df)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)