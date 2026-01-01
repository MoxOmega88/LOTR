# src/5_topic_modeling.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json

print("=" * 60)
print("📚 LOTR TOPIC MODELING")
print("=" * 60)

# Load data
df = pd.read_csv('../outputs/lotr_with_sentiment.csv')
print(f"📚 Loaded {len(df)} dialog lines")

# Prepare text data
print("\n🔄 Preparing text data...")
documents = df['dialog'].tolist()

# Create TF-IDF matrix
print("🔄 Creating TF-IDF matrix...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    max_df=0.8,
    min_df=5
)
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"✅ Created matrix with {len(feature_names)} features")

# Train LDA model
print("\n🔄 Training topic model...")
n_topics = 8
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20
)
lda_topics = lda_model.fit_transform(tfidf_matrix)

print(f"✅ Identified {n_topics} topics!")

# Display topics
print("\n📊 DISCOVERED TOPICS")
print("=" * 60)

def display_topics(model, feature_names, no_top_words=10):
    topics_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-no_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics_dict[f"Topic {topic_idx + 1}"] = top_words
        
        print(f"\nTopic {topic_idx + 1}:")
        print(f"  {', '.join(top_words)}")
    
    return topics_dict

topics_dict = display_topics(lda_model, feature_names)

# Assign topic labels manually based on keywords
topic_labels = {
    'Topic 1': 'War & Battle',
    'Topic 2': 'Friendship & Loyalty',
    'Topic 3': 'The Ring & Power',
    'Topic 4': 'Journey & Quest',
    'Topic 5': 'Fear & Danger',
    'Topic 6': 'Hope & Courage',
    'Topic 7': 'Death & Sacrifice',
    'Topic 8': 'Home & Peace'
}

# Add dominant topic to dataframe
dominant_topics = lda_topics.argmax(axis=1)
df['dominant_topic'] = [f"Topic {t+1}" for t in dominant_topics]
df['topic_label'] = df['dominant_topic'].map(topic_labels)

# Analyze topic distribution
print("\n📊 TOPIC DISTRIBUTION")
print("-" * 60)
topic_counts = df['topic_label'].value_counts()
print(topic_counts)

# Visualize topic distribution
plt.figure(figsize=(12, 6))
topic_counts.plot(kind='bar', color='steelblue')
plt.title('Topic Distribution in LOTR Dialogs', fontsize=16, fontweight='bold')
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Number of Dialogs', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../outputs/topic_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: topic_distribution.png")
plt.close()

# Topics by movie
print("\n🎬 TOPICS BY MOVIE")
print("-" * 60)
movie_topics = pd.crosstab(df['movie'], df['topic_label'])
print(movie_topics)

# Visualize
movie_topics_pct = movie_topics.div(movie_topics.sum(axis=1), axis=0) * 100
ax = movie_topics_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')
plt.title('Topic Distribution by Movie', fontsize=16, fontweight='bold')
plt.xlabel('Movie', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../outputs/topics_by_movie.png', dpi=300, bbox_inches='tight')
print("✅ Saved: topics_by_movie.png")
plt.close()

# Topics by character
print("\n👥 TOPICS BY CHARACTER (Top 8)")
print("-" * 60)
top_chars = df['char'].value_counts().head(8).index
df_top = df[df['char'].isin(top_chars)]

char_topics = pd.crosstab(df_top['char'], df_top['topic_label'])
char_topics_pct = char_topics.div(char_topics.sum(axis=1), axis=0) * 100

print(char_topics)

# Visualize
fig, ax = plt.subplots(figsize=(14, 8))
char_topics_pct.plot(kind='barh', stacked=True, ax=ax, colormap='tab10')
plt.title('Topic Distribution by Character', fontsize=16, fontweight='bold')
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Character', fontsize=12)
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../outputs/topics_by_character.png', dpi=300, bbox_inches='tight')
print("✅ Saved: topics_by_character.png")
plt.close()

# Create word clouds for each topic
print("\n☁️ GENERATING WORD CLOUDS FOR TOPICS...")
for topic_num in range(n_topics):
    topic_words = dict(zip(
        feature_names,
        lda_model.components_[topic_num]
    ))
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(topic_words)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Topic {topic_num + 1}: {topic_labels[f"Topic {topic_num + 1}"]}',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'../outputs/wordcloud_topic_{topic_num + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("✅ Saved 8 word cloud images")

# Save results - FIXED VERSION
df.to_csv('../outputs/lotr_with_topics.csv', index=False)
print("\n✅ Saved: lotr_with_topics.csv")

print("\n💾 Saving topic modeling JSON...")

# Clean the data
topics_dict_clean = {}
for topic_name, words in topics_dict.items():
    topics_dict_clean[topic_name] = [str(word) for word in words]

topic_results = {
    'topics': topics_dict_clean,
    'topic_labels': {k: str(v) for k, v in topic_labels.items()},
    'topic_distribution': {str(k): int(v) for k, v in topic_counts.to_dict().items()},
    'topics_by_movie': {
        str(k): {str(k2): int(v2) for k2, v2 in v.items()}
        for k, v in movie_topics.to_dict().items()
    },
    'topics_by_character': {
        str(k): {str(k2): int(v2) for k2, v2 in v.items()}
        for k, v in char_topics.to_dict().items()
    }
}

with open('../outputs/topic_modeling_results.json', 'w', encoding='utf-8') as f:
    json.dump(topic_results, f, indent=2, ensure_ascii=False)

print("✅ Saved: topic_modeling_results.json")

# Verify
with open('../outputs/topic_modeling_results.json', 'r') as f:
    test_load = json.load(f)
    print(f"✅ Verified: JSON is valid with {len(test_load['topics'])} topics")