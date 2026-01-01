# src/7_verify_json.py
import json
import os

print("=" * 60)
print("🔍 VERIFYING JSON FILES")
print("=" * 60)

files_to_check = [
    '../web-app/data/sentiment_summary.json',
    '../web-app/data/topic_modeling_results.json',
    '../web-app/data/quiz_data.json',
    '../web-app/data/similarity_search_results.json'
]

for file_path in files_to_check:
    print(f"\n📄 Checking: {file_path}")
    print("-" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ FILE NOT FOUND!")
        continue
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Valid JSON")
        print(f"📊 Top-level keys: {list(data.keys())[:10]}")
        
        # Check sentiment structure
        if 'sentiment' in file_path:
            if 'character_sentiment' in data:
                print(f"✅ Has 'character_sentiment' key")
                print(f"   Characters: {len(data['character_sentiment'])}")
                if len(data['character_sentiment']) > 0:
                    print(f"   Sample character: {data['character_sentiment'][0]}")
            else:
                print(f"⚠️  Missing 'character_sentiment' key")
                print(f"   Available keys: {list(data.keys())}")
        
        # Check topics structure
        if 'topic' in file_path:
            if 'topics' in data:
                print(f"✅ Has 'topics' key")
                print(f"   Number of topics: {len(data['topics'])}")
                if 'topic_labels' in data:
                    print(f"✅ Has 'topic_labels' key")
                else:
                    print(f"⚠️  Missing 'topic_labels' key")
            else:
                print(f"⚠️  Missing 'topics' key")
                print(f"   Available keys: {list(data.keys())}")
        
    except json.JSONDecodeError as e:
        print(f"❌ INVALID JSON: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n" + "=" * 60)
print("✅ VERIFICATION COMPLETE")
print("=" * 60)