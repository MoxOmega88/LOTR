# web-app/api.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load data and model
print("Loading model and data...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv(os.path.join(BASE_DIR, '../outputs/lotr_with_sentiment.csv'))
embeddings = np.load(os.path.join(BASE_DIR, '../outputs/dialog_embeddings.npy'))
print(f"✅ Loaded {len(df)} dialogs and embeddings")

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/search', methods=['POST'])
def search_dialogs():
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
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
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'total_dialogs': len(df)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)