# src/1_data_exploration.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create outputs folder
Path('../outputs').mkdir(exist_ok=True)

print("=" * 60)
print("📊 LOTR Dataset Exploration")
print("=" * 60)

# Load the dataset
df = pd.read_csv('../data/lotr_scripts.csv')

print("\n1️⃣ BASIC INFO")
print("-" * 60)
print(f"Total dialog lines: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

print("\n2️⃣ DATA QUALITY CHECK")
print("-" * 60)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

print("\n3️⃣ CHARACTER ANALYSIS")
print("-" * 60)
char_counts = df['char'].value_counts().head(15)
print(f"Top 15 characters by dialog count:\n{char_counts}")

# Visualize top characters
plt.figure(figsize=(14, 6))
char_counts.plot(kind='bar', color='steelblue')
plt.title('Top 15 Characters by Number of Dialog Lines', fontsize=16, fontweight='bold')
plt.xlabel('Character', fontsize=12)
plt.ylabel('Number of Lines', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../outputs/character_dialog_counts.png', dpi=300, bbox_inches='tight')
print("✅ Saved: character_dialog_counts.png")
plt.close()

print("\n4️⃣ MOVIE DISTRIBUTION")
print("-" * 60)
movie_counts = df['movie'].value_counts()
print(f"Dialog lines per movie:\n{movie_counts}")

# Visualize movie distribution
plt.figure(figsize=(10, 6))
movie_counts.plot(kind='bar', color=['#FFD700', '#C0C0C0', '#CD7F32'])
plt.title('Dialog Lines per Movie', fontsize=16, fontweight='bold')
plt.xlabel('Movie', fontsize=12)
plt.ylabel('Number of Lines', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../outputs/movie_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: movie_distribution.png")
plt.close()

print("\n5️⃣ DIALOG LENGTH ANALYSIS")
print("-" * 60)
df['dialog_length'] = df['dialog'].str.len()
df['word_count'] = df['dialog'].str.split().str.len()

print(f"Average dialog length: {df['dialog_length'].mean():.1f} characters")
print(f"Average word count: {df['word_count'].mean():.1f} words")
print(f"Longest dialog: {df['dialog_length'].max()} characters")
print(f"Shortest dialog: {df['dialog_length'].min()} characters")

# Find longest dialogs
print("\n📜 Longest Dialogs:")
longest = df.nlargest(3, 'dialog_length')[['char', 'dialog', 'dialog_length']]
for idx, row in longest.iterrows():
    print(f"\n{row['char']} ({row['dialog_length']} chars):")
    print(f"  {row['dialog'][:150]}...")

# Visualize dialog lengths by character
top_chars = df['char'].value_counts().head(10).index
df_top = df[df['char'].isin(top_chars)]

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_top, x='char', y='word_count', palette='Set2')
plt.title('Dialog Word Count Distribution by Character (Top 10)', fontsize=16, fontweight='bold')
plt.xlabel('Character', fontsize=12)
plt.ylabel('Word Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../outputs/character_wordcount_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: character_wordcount_distribution.png")
plt.close()

print("\n6️⃣ CLEAN DATA")
print("-" * 60)
# Remove any rows with missing dialog
df_clean = df.dropna(subset=['dialog'])
df_clean = df_clean[df_clean['dialog'].str.strip() != '']
print(f"Rows after cleaning: {len(df_clean)}")

# Save cleaned dataset
df_clean.to_csv('../data/lotr_scripts_clean.csv', index=False)
print("✅ Saved: lotr_scripts_clean.csv")

print("\n" + "=" * 60)
print("✅ DATA EXPLORATION COMPLETE!")
print("=" * 60)
print("\nNext step: Run 2_sentiment_analysis.py")