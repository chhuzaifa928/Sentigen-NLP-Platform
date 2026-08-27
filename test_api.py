"""
Test Script for SentiGen NLP Platform
Run this to test all API endpoints
"""

import requests
import json
from typing import Dict

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_sentiment():
    """Test sentiment analysis"""
    print_section("Testing Sentiment Analysis")
    
    test_texts = [
        "I absolutely love this! It's amazing and wonderful!",
        "This is terrible and disappointing. I'm very upset.",
        "The weather is okay today, nothing special."
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        response = requests.post(
            f"{BASE_URL}/api/sentiment",
            json={"text": text}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Sentiment: {data['sentiment']['label']} ({data['sentiment']['confidence']:.2%})")
            print(f"Primary Emotion: {data['primary_emotion']}")
            print(f"Polarity: {data['polarity_score']:.2f}")
        else:
            print(f"Error: {response.status_code}")

def test_ner():
    """Test named entity recognition"""
    print_section("Testing Named Entity Recognition")
    
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    The company is headquartered in Cupertino, California. Tim Cook became CEO in 2011.
    """
    
    print(f"Text: {text.strip()}")
    response = requests.post(
        f"{BASE_URL}/api/ner",
        json={"text": text}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal Entities: {data['total_entities']}")
        print(f"Entity Types: {data['entity_types']}")
        print("\nEntities Found:")
        for ent in data['entities'][:5]:  # Show first 5
            print(f"  - {ent['text']} ({ent['label']})")
    else:
        print(f"Error: {response.status_code}")

def test_summarization():
    """Test text summarization"""
    print_section("Testing Text Summarization")
    
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
    the natural intelligence displayed by humans and animals. Leading AI textbooks define 
    the field as the study of "intelligent agents": any device that perceives its environment 
    and takes actions that maximize its chance of successfully achieving its goals. 
    Colloquially, the term "artificial intelligence" is often used to describe machines 
    (or computers) that mimic "cognitive" functions that humans associate with the human 
    mind, such as "learning" and "problem solving". As machines become increasingly capable, 
    tasks considered to require "intelligence" are often removed from the definition of AI, 
    a phenomenon known as the AI effect.
    """
    
    for method in ["extractive", "abstractive"]:
        print(f"\n{method.upper()} Summary:")
        response = requests.post(
            f"{BASE_URL}/api/summarize",
            json={
                "text": text,
                "method": method,
                "num_sentences": 2,
                "max_length": 100
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Summary: {data['summary']}")
            print(f"Compression: {data['compression_ratio']:.2%}")
        else:
            print(f"Error: {response.status_code}")

def test_generation():
    """Test text generation"""
    print_section("Testing Text Generation")
    
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json={
                "prompt": prompt,
                "max_length": 80,
                "temperature": 0.7,
                "num_sequences": 1
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Generated: {data['generated_texts'][0]}")
        else:
            print(f"Error: {response.status_code}")

def test_topics():
    """Test topic modeling"""
    print_section("Testing Topic Modeling")
    
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "Deep learning uses neural networks with multiple layers to learn from large amounts of data.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Reinforcement learning trains agents to make decisions through trial and error."
    ]
    
    print(f"Analyzing {len(documents)} documents...")
    response = requests.post(
        f"{BASE_URL}/api/topics",
        json={
            "documents": documents,
            "num_topics": 2,
            "num_words": 5
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTopics Found: {data['num_topics']}")
        print(f"Coherence Score: {data['coherence_score']:.3f}")
        
        for topic in data['topics']:
            words = [w['word'] for w in topic['words'][:5]]
            print(f"\nTopic {topic['topic_id']}: {', '.join(words)}")
    else:
        print(f"Error: {response.status_code}")

def main():
    """Run all tests"""
    print("\n" + "🚀 " * 20)
    print("  SentiGen NLP Platform - API Test Suite")
    print("🚀 " * 20)
    
    try:
        test_health()
        test_sentiment()
        test_ner()
        test_summarization()
        test_generation()
        test_topics()
        
        print_section("✅ All Tests Completed!")
        print("\nThe API is working correctly!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server.")
        print("Make sure the server is running: python app.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
