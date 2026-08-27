"""
Example Usage Script
Demonstrates how to use each NLP module independently
"""

import sys
import os

# Add models directory to path
sys.path.append(os.path.dirname(__file__))

from models import (
    SentimentAnalyzer,
    NERExtractor,
    TextSummarizer,
    TextGenerator,
    TopicModeler
)

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS DEMO")
    print("="*60)
    
    analyzer = SentimentAnalyzer()
    
    texts = [
        "I absolutely love this! It's amazing!",
        "This is terrible and disappointing.",
        "The product is okay, nothing special."
    ]
    
    for text in texts:
        print(f"\nText: '{text}'")
        result = analyzer.analyze_sentiment(text)
        print(f"Sentiment: {result['sentiment']['label']} ({result['sentiment']['confidence']:.2%})")
        print(f"Primary Emotion: {result['primary_emotion']}")
        print(f"Polarity Score: {result['polarity_score']:.2f}")

def demo_ner():
    """Demonstrate named entity recognition"""
    print("\n" + "="*60)
    print("NAMED ENTITY RECOGNITION DEMO")
    print("="*60)
    
    extractor = NERExtractor()
    
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne 
    in April 1976 in Cupertino, California. Tim Cook became CEO in 2011.
    """
    
    print(f"\nText: {text.strip()}")
    result = extractor.extract_entities(text)
    
    print(f"\nTotal Entities: {result['total_entities']}")
    print("\nEntities Found:")
    for ent in result['entities']:
        print(f"  - {ent['text']} ({ent['label']}): {ent['description']}")

def demo_summarization():
    """Demonstrate text summarization"""
    print("\n" + "="*60)
    print("TEXT SUMMARIZATION DEMO")
    print("="*60)
    
    summarizer = TextSummarizer()
    
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines (or computers) 
    that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving". As machines become increasingly 
    capable, tasks considered to require "intelligence" are often removed from 
    the definition of AI, a phenomenon known as the AI effect.
    """
    
    print("\nOriginal Text:")
    print(text.strip())
    
    # Extractive summary
    print("\n--- EXTRACTIVE SUMMARY ---")
    result = summarizer.summarize(text, method="extractive", num_sentences=2)
    print(result['summary'])
    print(f"Compression: {result['compression_ratio']:.2%}")
    
    # Abstractive summary
    print("\n--- ABSTRACTIVE SUMMARY ---")
    result = summarizer.summarize(text, method="abstractive", max_length=100)
    print(result['summary'])
    print(f"Compression: {result['compression_ratio']:.2%}")

def demo_text_generation():
    """Demonstrate text generation"""
    print("\n" + "="*60)
    print("TEXT GENERATION DEMO")
    print("="*60)
    
    generator = TextGenerator()
    
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        result = generator.generate(
            prompt,
            max_length=80,
            temperature=0.7,
            num_return_sequences=2
        )
        
        for i, text in enumerate(result['generated_texts'], 1):
            print(f"\nVariation {i}:")
            print(text)

def demo_topic_modeling():
    """Demonstrate topic modeling"""
    print("\n" + "="*60)
    print("TOPIC MODELING DEMO")
    print("="*60)
    
    modeler = TopicModeler()
    
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "Deep learning uses neural networks with multiple layers to learn from large amounts of data.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Supervised learning uses labeled data to train models for prediction tasks.",
        "Unsupervised learning finds patterns in unlabeled data without explicit guidance."
    ]
    
    print(f"\nAnalyzing {len(documents)} documents...")
    result = modeler.discover_topics(documents, num_topics=3, num_words=5)
    
    print(f"\nTopics Found: {result['num_topics']}")
    print(f"Coherence Score: {result['coherence_score']:.3f}")
    
    for topic in result['topics']:
        words = [w['word'] for w in topic['words'][:5]]
        print(f"\nTopic {topic['topic_id']}: {topic['label']}")
        print(f"  Words: {', '.join(words)}")

def main():
    """Run all demos"""
    print("\n" + "🚀 " * 20)
    print("  SentiGen NLP Platform - Example Usage")
    print("🚀 " * 20)
    
    try:
        # Run each demo
        demo_sentiment_analysis()
        demo_ner()
        demo_summarization()
        demo_text_generation()
        demo_topic_modeling()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
