# src/3_character_quiz.py
import pandas as pd
import json
from collections import Counter

print("=" * 60)
print("🎯 LOTR CHARACTER QUIZ BUILDER")
print("=" * 60)

# Load sentiment data
df = pd.read_csv('../outputs/lotr_with_sentiment.csv')

# Define character profiles based on actual data
print("\n🔍 Analyzing character personalities from dialog...")

# Get top 12 main characters
main_characters = df['char'].value_counts().head(12).index.tolist()

character_profiles = {}

for char in main_characters:
    char_data = df[df['char'] == char]
    
    # Calculate sentiment ratio
    positive_ratio = (char_data['sentiment'] == 'POSITIVE').sum() / len(char_data)
    
    # Average dialog length (indicates verbosity)
    avg_words = char_data['dialog'].str.split().str.len().mean()
    
    # Determine traits based on data
    traits = []
    
    # Sentiment-based traits
    if positive_ratio > 0.65:
        traits.append('optimistic')
    elif positive_ratio < 0.45:
        traits.append('serious')
    else:
        traits.append('balanced')
    
    # Dialog style traits
    if avg_words > 20:
        traits.append('wise')
        traits.append('eloquent')
    elif avg_words < 10:
        traits.append('direct')
        traits.append('action-oriented')
    else:
        traits.append('thoughtful')
    
    # Character-specific traits (manual enhancement)
    char_specific = {
        'GANDALF': ['wise', 'powerful', 'protective', 'mysterious'],
        'FRODO': ['brave', 'burdened', 'determined', 'compassionate'],
        'SAM': ['loyal', 'brave', 'optimistic', 'humble', 'devoted'],
        'ARAGORN': ['noble', 'leader', 'courageous', 'humble'],
        'LEGOLAS': ['graceful', 'skilled', 'observant', 'loyal'],
        'GIMLI': ['fierce', 'loyal', 'stubborn', 'humorous'],
        'PIPPIN': ['curious', 'brave', 'mischievous', 'loyal'],
        'MERRY': ['clever', 'brave', 'loyal', 'humorous'],
        'BOROMIR': ['brave', 'conflicted', 'protective', 'noble'],
        'GOLLUM': ['obsessive', 'conflicted', 'cunning', 'tragic'],
        'SARUMAN': ['ambitious', 'manipulative', 'powerful', 'fallen'],
        'GALADRIEL': ['wise', 'powerful', 'ethereal', 'protective']
    }
    
    if char in char_specific:
        traits = char_specific[char]
    
    # Get iconic quote
    char_dialogs = char_data.nlargest(5, 'confidence')
    iconic_quote = char_dialogs.iloc[0]['dialog'] if len(char_dialogs) > 0 else "..."
    
    character_profiles[char] = {
        'traits': traits,
        'quote': iconic_quote[:100] + '...' if len(iconic_quote) > 100 else iconic_quote,
        'positive_ratio': round(positive_ratio, 2),
        'avg_words': round(avg_words, 1),
        'total_lines': int(len(char_data))
    }

print("\n✅ Character profiles created!")

# Create quiz questions
quiz_questions = [
    {
        'id': 1,
        'question': 'When faced with overwhelming danger, you:',
        'options': [
            {'text': 'Face it head-on, no matter the cost', 'traits': ['brave', 'courageous', 'determined']},
            {'text': 'Seek wisdom before acting', 'traits': ['wise', 'thoughtful', 'strategic']},
            {'text': 'Stand by your friends, supporting them', 'traits': ['loyal', 'devoted', 'protective']},
            {'text': 'Use cunning and deception', 'traits': ['cunning', 'clever', 'manipulative']}
        ]
    },
    {
        'id': 2,
        'question': 'Your greatest strength is:',
        'options': [
            {'text': 'My unwavering loyalty', 'traits': ['loyal', 'devoted', 'humble']},
            {'text': 'My wisdom and knowledge', 'traits': ['wise', 'eloquent', 'mysterious']},
            {'text': 'My combat skills', 'traits': ['skilled', 'action-oriented', 'fierce']},
            {'text': 'My determination', 'traits': ['determined', 'brave', 'serious']}
        ]
    },
    {
        'id': 3,
        'question': 'In a group, you are usually:',
        'options': [
            {'text': 'The wise advisor', 'traits': ['wise', 'thoughtful', 'protective']},
            {'text': 'The natural leader', 'traits': ['leader', 'noble', 'courageous']},
            {'text': 'The loyal companion', 'traits': ['loyal', 'humble', 'devoted']},
            {'text': 'The comic relief', 'traits': ['humorous', 'mischievous', 'curious']}
        ]
    },
    {
        'id': 4,
        'question': 'What motivates you most?',
        'options': [
            {'text': 'Protecting those I love', 'traits': ['protective', 'loyal', 'brave']},
            {'text': 'Fulfilling my destiny', 'traits': ['noble', 'determined', 'burdened']},
            {'text': 'Seeking power and knowledge', 'traits': ['ambitious', 'powerful', 'cunning']},
            {'text': 'Preserving the greater good', 'traits': ['wise', 'balanced', 'compassionate']}
        ]
    },
    {
        'id': 5,
        'question': 'Your speaking style is:',
        'options': [
            {'text': 'Wise and poetic', 'traits': ['wise', 'eloquent', 'ethereal']},
            {'text': 'Direct and to the point', 'traits': ['direct', 'action-oriented', 'fierce']},
            {'text': 'Warm and encouraging', 'traits': ['optimistic', 'compassionate', 'devoted']},
            {'text': 'Witty with humor', 'traits': ['humorous', 'clever', 'mischievous']}
        ]
    },
    {
        'id': 6,
        'question': 'When tempted by power, you:',
        'options': [
            {'text': 'Resist with all your might', 'traits': ['brave', 'determined', 'noble']},
            {'text': 'Feel the weight of the burden', 'traits': ['burdened', 'conflicted', 'serious']},
            {'text': 'Are not interested - loyalty matters more', 'traits': ['loyal', 'humble', 'devoted']},
            {'text': 'Would be tempted to use it for good', 'traits': ['ambitious', 'conflicted', 'fallen']}
        ]
    },
    {
        'id': 7,
        'question': 'Your ideal role in a quest:',
        'options': [
            {'text': 'The ringbearer - carrying the burden', 'traits': ['brave', 'burdened', 'determined']},
            {'text': 'The guide who shows the way', 'traits': ['wise', 'powerful', 'protective']},
            {'text': 'The faithful companion', 'traits': ['loyal', 'devoted', 'optimistic']},
            {'text': 'The skilled warrior', 'traits': ['skilled', 'graceful', 'fierce']}
        ]
    },
    {
        'id': 8,
        'question': 'How do you handle conflict?',
        'options': [
            {'text': 'With diplomatic wisdom', 'traits': ['wise', 'balanced', 'thoughtful']},
            {'text': 'With fierce determination', 'traits': ['fierce', 'stubborn', 'brave']},
            {'text': 'By supporting others', 'traits': ['loyal', 'protective', 'humble']},
            {'text': 'With cunning strategy', 'traits': ['clever', 'cunning', 'observant']}
        ]
    }
]

print("\n📝 Created 8 quiz questions with trait mapping")

# Create quiz logic function
def calculate_quiz_result(user_answers):
    """
    user_answers: list of traits from user selections
    Returns: matched character
    """
    # Count trait occurrences
    trait_counter = Counter()
    for answer_traits in user_answers:
        trait_counter.update(answer_traits)
    
    # Find best matching character
    best_match = None
    best_score = 0
    
    for char, profile in character_profiles.items():
        score = sum(trait_counter.get(trait, 0) for trait in profile['traits'])
        if score > best_score:
            best_score = score
            best_match = char
    
    return best_match, trait_counter

# Test the quiz
print("\n🧪 Testing quiz logic...")
test_answers = [
    ['brave', 'courageous', 'determined'],
    ['loyal', 'devoted', 'humble'],
    ['loyal', 'humble', 'devoted'],
    ['protective', 'loyal', 'brave'],
    ['warm', 'compassionate', 'devoted'],
    ['brave', 'determined', 'noble'],
    ['loyal', 'devoted', 'optimistic'],
    ['loyal', 'protective', 'humble']
]

result_char, traits = calculate_quiz_result(test_answers)
print(f"Test result: {result_char}")
print(f"Dominant traits: {traits.most_common(5)}")

# Export quiz data
quiz_export = {
    'questions': quiz_questions,
    'characters': {
        char: {
            **profile,
            'description': f"You embody {', '.join(profile['traits'][:3])}"
        }
        for char, profile in character_profiles.items()
    }
}

with open('../outputs/quiz_data.json', 'w') as f:
    json.dump(quiz_export, f, indent=2)

print("\n✅ Saved: quiz_data.json")

print("\n" + "=" * 60)
print("✅ CHARACTER QUIZ SYSTEM COMPLETE!")
print("=" * 60)
print("\nNext step: Run 4_similarity_search.py")