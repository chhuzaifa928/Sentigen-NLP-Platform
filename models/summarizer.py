"""
Text Summarization Module
Provides both extractive and abstractive summarization
"""

from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextSummarizer:
    def __init__(self):
        """Initialize summarization models"""
        try:
            logger.info("Loading summarization models...")
            
            # Abstractive summarization using T5
            self.abstractive_pipeline = pipeline(
                "summarization",
                model="t5-small"
            )
            
            self.stop_words = set(stopwords.words('english'))
            
            logger.info("Text summarizer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing summarizer: {e}")
            raise
    
    def summarize(self, text: str, method: str = "abstractive", 
                  num_sentences: int = 3, max_length: int = 150) -> Dict:
        """
        Summarize text using specified method
        
        Args:
            text: Input text to summarize
            method: 'extractive' or 'abstractive'
            num_sentences: Number of sentences for extractive summary
            max_length: Maximum length for abstractive summary
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            if method == "extractive":
                summary = self._extractive_summarize(text, num_sentences)
            elif method == "abstractive":
                summary = self._abstractive_summarize(text, max_length)
            else:
                summary = self._extractive_summarize(text, num_sentences)
            
            # Calculate compression ratio
            compression_ratio = len(summary) / len(text) if len(text) > 0 else 0
            
            return {
                "summary": summary,
                "method": method,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(compression_ratio, 4),
                "original_sentences": len(sent_tokenize(text)),
                "summary_sentences": len(sent_tokenize(summary))
            }
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return {
                "error": str(e),
                "summary": text[:200] + "..." if len(text) > 200 else text,
                "method": method,
                "original_length": len(text),
                "summary_length": 0,
                "compression_ratio": 0.0
            }
    
    def _extractive_summarize(self, text: str, num_sentences: int) -> str:
        """
        Extractive summarization using TF-IDF
        Selects the most important sentences from the original text
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # If TF-IDF fails, return first n sentences
            return ' '.join(sentences[:num_sentences])
        
        # Calculate sentence scores (sum of TF-IDF values)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get indices of top sentences
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        
        # Sort indices to maintain original order
        top_sentence_indices = sorted(top_sentence_indices)
        
        # Extract top sentences
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        
        return ' '.join(summary_sentences)
    
    def _abstractive_summarize(self, text: str, max_length: int) -> str:
        """
        Abstractive summarization using T5 transformer
        Generates new sentences that capture the essence of the text
        """
        # T5 has input length limitations
        max_input_length = 512
        
        # Truncate if necessary
        if len(text.split()) > max_input_length:
            words = text.split()[:max_input_length]
            text = ' '.join(words)
        
        # Calculate min_length as 30% of max_length, never exceeding max_length
        min_length = min(
            max(30, int(max_length * 0.3)),
            max(1, max_length // 2)
        )
        
        # Generate summary
        summary = self.abstractive_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return summary[0]['summary_text']
    
    def summarize_both(self, text: str, num_sentences: int = 3, 
                      max_length: int = 150) -> Dict:
        """Generate both extractive and abstractive summaries"""
        extractive = self.summarize(text, "extractive", num_sentences)
        abstractive = self.summarize(text, "abstractive", max_length=max_length)
        
        return {
            "extractive": extractive,
            "abstractive": abstractive,
            "comparison": {
                "extractive_length": extractive['summary_length'],
                "abstractive_length": abstractive['summary_length'],
                "extractive_compression": extractive['compression_ratio'],
                "abstractive_compression": abstractive['compression_ratio']
            }
        }
    
    def get_key_sentences(self, text: str, num_sentences: int = 5) -> List[Dict]:
        """Get ranked key sentences with scores"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return [{"sentence": s, "score": 1.0, "rank": i+1} 
                   for i, s in enumerate(sentences)]
        
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Normalize scores
            max_score = sentence_scores.max()
            if max_score > 0:
                sentence_scores = sentence_scores / max_score
            
            # Get top sentences with scores
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            
            key_sentences = [
                {
                    "sentence": sentences[idx],
                    "score": round(float(sentence_scores[idx]), 4),
                    "rank": rank + 1,
                    "position": idx
                }
                for rank, idx in enumerate(top_indices)
            ]
            
            return sorted(key_sentences, key=lambda x: x['score'], reverse=True)
            
        except ValueError:
            return [{"sentence": s, "score": 1.0, "rank": i+1} 
                   for i, s in enumerate(sentences[:num_sentences])]


# Test function
if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
    the natural intelligence displayed by humans and animals. Leading AI textbooks define 
    the field as the study of "intelligent agents": any device that perceives its environment 
    and takes actions that maximize its chance of successfully achieving its goals. 
    Colloquially, the term "artificial intelligence" is often used to describe machines 
    (or computers) that mimic "cognitive" functions that humans associate with the human 
    mind, such as "learning" and "problem solving". As machines become increasingly capable, 
    tasks considered to require "intelligence" are often removed from the definition of AI, 
    a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever 
    hasn't been done yet." For instance, optical character recognition is frequently excluded 
    from things considered to be AI, having become a routine technology.
    """
    
    print("=== Extractive Summary ===")
    result = summarizer.summarize(test_text, method="extractive", num_sentences=2)
    print(result['summary'])
    print(f"\nCompression: {result['compression_ratio']:.2%}")
    
    print("\n=== Abstractive Summary ===")
    result = summarizer.summarize(test_text, method="abstractive", max_length=100)
    print(result['summary'])
    print(f"\nCompression: {result['compression_ratio']:.2%}")
