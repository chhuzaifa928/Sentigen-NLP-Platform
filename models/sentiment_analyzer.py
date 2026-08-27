"""
Sentiment Analysis Module
Performs multi-aspect sentiment analysis using transformer models
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analysis models"""
        try:
            # Primary sentiment classifier
            logger.info("Loading sentiment analysis model...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Emotion detection model
            logger.info("Loading emotion detection model...")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            
            logger.info("Sentiment analyzer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Perform comprehensive sentiment analysis
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment and emotion analysis
        """
        try:
            # Basic sentiment
            sentiment_result = self.sentiment_pipeline(text)[0]
            
            # Emotion detection
            emotion_results = self.emotion_pipeline(text)[0]
            
            # Sort emotions by score
            emotions_sorted = sorted(emotion_results, key=lambda x: x['score'], reverse=True)
            
            # Calculate overall polarity score
            polarity = self._calculate_polarity(sentiment_result, emotions_sorted)
            
            return {
                "sentiment": {
                    "label": sentiment_result['label'],
                    "confidence": round(sentiment_result['score'], 4)
                },
                "emotions": [
                    {
                        "emotion": emotion['label'],
                        "score": round(emotion['score'], 4)
                    }
                    for emotion in emotions_sorted
                ],
                "primary_emotion": emotions_sorted[0]['label'],
                "polarity_score": polarity,
                "analysis": self._generate_analysis(sentiment_result, emotions_sorted)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "error": str(e),
                "sentiment": {"label": "UNKNOWN", "confidence": 0.0},
                "emotions": [],
                "primary_emotion": "unknown",
                "polarity_score": 0.0,
                "analysis": "Error occurred during analysis"
            }
    
    def _calculate_polarity(self, sentiment: Dict, emotions: List[Dict]) -> float:
        """Calculate overall polarity score (-1 to 1)"""
        # Base polarity from sentiment
        base_polarity = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
        
        # Adjust based on emotions
        emotion_weights = {
            'joy': 0.8,
            'love': 0.9,
            'surprise': 0.3,
            'neutral': 0.0,
            'sadness': -0.6,
            'anger': -0.8,
            'fear': -0.7,
            'disgust': -0.9
        }
        
        emotion_adjustment = sum(
            emotion_weights.get(e['label'], 0) * e['score'] 
            for e in emotions[:3]  # Top 3 emotions
        ) / 3
        
        # Combine base polarity and emotion adjustment
        final_polarity = (base_polarity * 0.6) + (emotion_adjustment * 0.4)
        
        return round(final_polarity, 4)
    
    def _generate_analysis(self, sentiment: Dict, emotions: List[Dict]) -> str:
        """Generate human-readable analysis"""
        sentiment_label = sentiment['label'].lower()
        primary_emotion = emotions[0]['label']
        emotion_confidence = emotions[0]['score']
        
        analysis = f"The text expresses a {sentiment_label} sentiment "
        analysis += f"with {primary_emotion} as the dominant emotion "
        analysis += f"(confidence: {emotion_confidence:.2%}). "
        
        # Add secondary emotions if significant
        significant_emotions = [e for e in emotions[1:3] if e['score'] > 0.1]
        if significant_emotions:
            emotion_names = ", ".join([e['label'] for e in significant_emotions])
            analysis += f"Secondary emotions detected: {emotion_names}."
        
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts"""
        return [self.analyze_sentiment(text) for text in texts]


# Test function
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I absolutely love this! It's amazing and wonderful!",
        "This is terrible and disappointing. I'm very upset.",
        "The weather is okay today, nothing special.",
        "I'm scared and worried about what might happen."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = analyzer.analyze_sentiment(text)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Primary Emotion: {result['primary_emotion']}")
        print(f"Polarity: {result['polarity_score']}")
        print(f"Analysis: {result['analysis']}")
