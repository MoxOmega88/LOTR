# src/2_sentiment_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("😊 LOTR SENTIMENT ANALYSIS")
print("=" * 60)

# Load cleaned data
df = pd.read_csv('../data/lotr_scripts_clean.csv')
print(f"📚 Loaded {len(df)} dialog lines")

# Initialize sentiment analyzer
print("\n🔄 Loading sentiment model (this may take a minute)...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # Use CPU
)
print("✅ Model loaded!")

# Analyze sentiments
print("\n🔄 Analyzing sentiments (this will take a few minutes)...")
sentiments = []
confidences = []

# Process in batches to speed up
batch_size = 32
for i in tqdm(range(0, len(df), batch_size)):
    batch = df['dialog'].iloc[i:i+batch_size].tolist()
    # Truncate long dialogs to avoid errors
    batch = [dialog[:512] for dialog in batch]
    
    results = sentiment_analyzer(batch)
    
    for result in results:
        sentiments.append(result['label'])
        confidences.append(result['score'])

df['sentiment'] = sentiments
df['confidence'] = confidences

print("✅ Sentiment analysis complete!")

# Calculate statistics
print("\n📊 OVERALL SENTIMENT DISTRIBUTION")
print("-" * 60)
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
print(f"\nPositive: {sentiment_counts.get('POSITIVE', 0) / len(df) * 100:.1f}%")
print(f"Negative: {sentiment_counts.get('NEGATIVE', 0) / len(df) * 100:.1f}%")

# Visualize overall sentiment
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c']
sentiment_counts.plot(kind='bar', color=colors)
plt.title('Overall Sentiment Distribution in LOTR', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Lines', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../outputs/overall_sentiment.png', dpi=300, bbox_inches='tight')
print("✅ Saved: overall_sentiment.png")
plt.close()

# Character-level sentiment analysis
print("\n😊 CHARACTER SENTIMENT ANALYSIS")
print("-" * 60)

# Get top 10 characters
top_chars = df['char'].value_counts().head(15).index
df_top = df[df['char'].isin(top_chars)]

char_sentiment = []
for char in top_chars:
    char_data = df_top[df_top['char'] == char]
    positive = (char_data['sentiment'] == 'POSITIVE').sum()
    negative = (char_data['sentiment'] == 'NEGATIVE').sum()
    total = len(char_data)
    
    char_sentiment.append({
        'Character': char,
        'Total Lines': total,
        'Positive': positive,
        'Negative': negative,
        'Positive %': round(positive / total * 100, 1),
        'Negative %': round(negative / total * 100, 1),
        'Avg Confidence': round(char_data['confidence'].mean(), 2)
    })

char_df = pd.DataFrame(char_sentiment)
print(char_df.to_string(index=False))

# Visualize character sentiments
fig, ax = plt.subplots(figsize=(14, 8))
x = range(len(char_df))
width = 0.35

ax.bar([i - width/2 for i in x], char_df['Positive %'], width, 
       label='Positive', color='#2ecc71', alpha=0.8)
ax.bar([i + width/2 for i in x], char_df['Negative %'], width, 
       label='Negative', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Character', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Sentiment Distribution by Character (Top 10)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(char_df['Character'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/character_sentiment_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: character_sentiment_comparison.png")
plt.close()

# Movie sentiment comparison
print("\n🎬 SENTIMENT BY MOVIE")
print("-" * 60)
movie_sentiment = df.groupby(['movie', 'sentiment']).size().unstack(fill_value=0)
movie_sentiment_pct = movie_sentiment.div(movie_sentiment.sum(axis=1), axis=0) * 100
print(movie_sentiment_pct)

# Visualize
movie_sentiment_pct.plot(kind='bar', figsize=(12, 6), color=['#e74c3c', '#2ecc71'])
plt.title('Sentiment Distribution by Movie', fontsize=16, fontweight='bold')
plt.xlabel('Movie', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('../outputs/movie_sentiment.png', dpi=300, bbox_inches='tight')
print("✅ Saved: movie_sentiment.png")
plt.close()

# Find most positive and negative dialogs
print("\n🌟 MOST POSITIVE DIALOGS")
print("-" * 60)
positive_df = df[df['sentiment'] == 'POSITIVE'].nlargest(5, 'confidence')
for idx, row in positive_df.iterrows():
    print(f"\n{row['char']} (confidence: {row['confidence']:.2f}):")
    print(f"  {row['dialog'][:150]}...")

print("\n💔 MOST NEGATIVE DIALOGS")
print("-" * 60)
negative_df = df[df['sentiment'] == 'NEGATIVE'].nlargest(5, 'confidence')
for idx, row in negative_df.iterrows():
    print(f"\n{row['char']} (confidence: {row['confidence']:.2f}):")
    print(f"  {row['dialog'][:150]}...")

# Save results
df.to_csv('../outputs/lotr_with_sentiment.csv', index=False)
print("\n✅ Saved: lotr_with_sentiment.csv")

# Save summary as JSON - FIXED VERSION
print("\n💾 Saving sentiment summary JSON...")

# Ensure char_sentiment is properly formatted
char_sentiment_clean = []
for item in char_sentiment:
    char_sentiment_clean.append({
        'Character': str(item['Character']),
        'Total Lines': int(item['Total Lines']),
        'Positive': int(item['Positive']),
        'Negative': int(item['Negative']),
        'Positive %': float(item['Positive %']),
        'Negative %': float(item['Negative %']),
        'Avg Confidence': float(item['Avg Confidence'])
    })

summary = {
    'overall_sentiment': {
        'POSITIVE': int(sentiment_counts.get('POSITIVE', 0)),
        'NEGATIVE': int(sentiment_counts.get('NEGATIVE', 0))
    },
    'character_sentiment': char_sentiment_clean,
    'movie_sentiment': {
        'POSITIVE': {k: float(v) for k, v in movie_sentiment_pct['POSITIVE'].items()},
        'NEGATIVE': {k: float(v) for k, v in movie_sentiment_pct['NEGATIVE'].items()}
    }
}

# Write with explicit encoding
with open('../outputs/sentiment_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ Saved: sentiment_summary.json")

# Verify it was written correctly
with open('../outputs/sentiment_summary.json', 'r') as f:
    test_load = json.load(f)
    print(f"✅ Verified: JSON is valid with {len(test_load['character_sentiment'])} characters")