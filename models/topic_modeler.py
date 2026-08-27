"""
Topic Modeling Module
Discovers hidden topics in documents using LDA
"""

from gensim import corpora, models
from gensim.models import CoherenceModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
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


class TopicModeler:
    def __init__(self):
        """Initialize topic modeling"""
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        logger.info("Topic modeler initialized successfully!")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for topic modeling"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        tokens = [
            token for token in tokens 
            if token not in self.stop_words 
            and token not in self.punctuation
            and len(token) > 2
        ]
        
        return tokens
    
    def discover_topics(self, documents: List[str], num_topics: int = 5, 
                       num_words: int = 10) -> Dict:
        """
        Discover topics in a collection of documents
        
        Args:
            documents: List of text documents
            num_topics: Number of topics to discover
            num_words: Number of words per topic
            
        Returns:
            Dictionary containing topics and analysis
        """
        try:
            if len(documents) < 2:
                return {
                    "error": "Need at least 2 documents for topic modeling",
                    "topics": [],
                    "num_topics": 0
                }
            
            # Preprocess documents
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            
            # Filter out empty documents
            processed_docs = [doc for doc in processed_docs if len(doc) > 0]
            
            if len(processed_docs) < 2:
                return {
                    "error": "Not enough valid documents after preprocessing",
                    "topics": [],
                    "num_topics": 0
                }
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(processed_docs)
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            
            # Adjust num_topics if necessary
            num_topics = min(num_topics, len(processed_docs) - 1)
            
            # Build LDA model
            lda_model = models.LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                per_word_topics=True
            )
            
            # Extract topics
            topics = []
            for idx, topic in lda_model.print_topics(num_words=num_words):
                # Parse topic string
                words = []
                for word_prob in topic.split(' + '):
                    prob, word = word_prob.split('*')
                    words.append({
                        "word": word.strip('"'),
                        "probability": float(prob)
                    })
                
                topics.append({
                    "topic_id": idx,
                    "words": words,
                    "label": self._generate_topic_label(words)
                })
            
            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=processed_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            
            # Get document-topic distribution
            doc_topics = self._get_document_topics(lda_model, corpus, documents)
            
            return {
                "topics": topics,
                "num_topics": num_topics,
                "num_documents": len(documents),
                "coherence_score": round(coherence_score, 4),
                "document_topics": doc_topics,
                "vocabulary_size": len(dictionary)
            }
            
        except Exception as e:
            logger.error(f"Error discovering topics: {e}")
            return {
                "error": str(e),
                "topics": [],
                "num_topics": 0,
                "num_documents": len(documents)
            }
    
    def _generate_topic_label(self, words: List[Dict]) -> str:
        """Generate a human-readable label for a topic"""
        # Use top 3 words
        top_words = [w['word'] for w in words[:3]]
        return " | ".join(top_words)
    
    def _get_document_topics(self, lda_model, corpus, documents) -> List[Dict]:
        """Get topic distribution for each document"""
        doc_topics = []
        
        for idx, (doc, doc_bow) in enumerate(zip(documents, corpus)):
            # Get topic distribution for this document
            topic_dist = lda_model.get_document_topics(doc_bow)
            
            # Sort by probability
            topic_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)
            
            doc_topics.append({
                "document_id": idx,
                "preview": doc[:100] + "..." if len(doc) > 100 else doc,
                "topics": [
                    {
                        "topic_id": topic_id,
                        "probability": round(prob, 4)
                    }
                    for topic_id, prob in topic_dist
                ]
            })
        
        return doc_topics
    
    def find_similar_documents(self, documents: List[str], 
                              query_doc: str, top_n: int = 5) -> List[Dict]:
        """Find documents similar to a query document"""
        try:
            # Preprocess all documents
            all_docs = documents + [query_doc]
            processed_docs = [self.preprocess_text(doc) for doc in all_docs]
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(processed_docs)
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            
            # Build LDA model
            lda_model = models.LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=min(5, len(documents)),
                random_state=42
            )
            
            # Get query document topics
            query_bow = corpus[-1]
            query_topics = lda_model.get_document_topics(query_bow)
            query_vec = dict(query_topics)
            
            # Calculate similarity with other documents
            similarities = []
            for idx, doc_bow in enumerate(corpus[:-1]):
                doc_topics = lda_model.get_document_topics(doc_bow)
                doc_vec = dict(doc_topics)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vec, doc_vec)
                
                similarities.append({
                    "document_id": idx,
                    "document": documents[idx][:200] + "..." if len(documents[idx]) > 200 else documents[idx],
                    "similarity": round(similarity, 4)
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities[:top_n]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate cosine similarity between two topic vectors"""
        # Get all topic IDs
        all_topics = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_topics)
        mag1 = sum(v**2 for v in vec1.values()) ** 0.5
        mag2 = sum(v**2 for v in vec2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


# Test function
if __name__ == "__main__":
    modeler = TopicModeler()
    
    test_documents = [
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "Deep learning uses neural networks with multiple layers to learn from large amounts of data.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Supervised learning uses labeled data to train models for prediction tasks.",
        "Unsupervised learning finds patterns in unlabeled data without explicit guidance."
    ]
    
    result = modeler.discover_topics(test_documents, num_topics=3, num_words=5)
    
    print(f"Number of Topics: {result['num_topics']}")
    print(f"Coherence Score: {result['coherence_score']}")
    print(f"\nTopics:")
    
    for topic in result['topics']:
        print(f"\nTopic {topic['topic_id']}: {topic['label']}")
        print("Words:")
        for word in topic['words']:
            print(f"  - {word['word']}: {word['probability']:.4f}")
