"""
Topic Modeling Module
Discovers hidden topics in documents using sklearn's Latent Dirichlet Allocation
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import string
import math
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
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

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for topic modeling"""
        # Lowercase and remove stopwords + punctuation
        words = word_tokenize_text(text, self.stop_words, self.punctuation)
        return ' '.join(words)

    def discover_topics(self, documents: List[str], num_topics: int = 5,
                        num_words: int = 10) -> Dict:
        """
        Discover topics in a collection of documents using LDA

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
            valid_pairs = [(d, p) for d, p in zip(documents, processed_docs) if len(p.split()) > 0]
            if len(valid_pairs) < 2:
                return {
                    "error": "Not enough valid documents after preprocessing",
                    "topics": [],
                    "num_topics": 0
                }

            docs, processed = zip(*valid_pairs)

            # Build vocabulary
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=1,
                max_features=10000,
                token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
            doc_term_matrix = vectorizer.fit_transform(processed)

            if doc_term_matrix.shape[0] < 2:
                return {
                    "error": "Not enough valid documents after preprocessing",
                    "topics": [],
                    "num_topics": 0
                }

            # Adjust num_topics if necessary
            num_topics = min(num_topics, doc_term_matrix.shape[0] - 1)

            # Build LDA model
            lda_model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=20,
                learning_method='batch'
            )
            lda_model.fit(doc_term_matrix)

            feature_names = vectorizer.get_feature_names_out()
            vocab_size = len(feature_names)

            # Extract topics
            topics = []
            for idx, topic_dist in enumerate(lda_model.components_):
                # Get top words for this topic
                top_word_indices = topic_dist.argsort()[::-1][:num_words]
                words = [
                    {
                        "word": feature_names[word_idx],
                        "probability": round(float(topic_dist[word_idx] / topic_dist.sum()), 4)
                    }
                    for word_idx in top_word_indices
                ]
                topics.append({
                    "topic_id": idx,
                    "words": words,
                    "label": self._generate_topic_label(words)
                })

            # Compute document-topic distribution
            doc_topics = self._get_document_topics(lda_model, doc_term_matrix, docs)

            # Compute coherence score
            word_to_idx = {word: i for i, word in enumerate(feature_names)}
            coherence_score = self._compute_coherence(
                doc_term_matrix, word_to_idx, topics, num_words
            )

            return {
                "topics": topics,
                "num_topics": num_topics,
                "num_documents": len(docs),
                "coherence_score": coherence_score,
                "document_topics": doc_topics,
                "vocabulary_size": vocab_size
            }

        except Exception as e:
            logger.error(f"Error discovering topics: {e}")
            return {
                "error": str(e),
                "topics": [],
                "num_topics": 0,
                "num_documents": len(documents)
            }

    def _compute_coherence(self, doc_term_matrix, word_to_idx, topics: List[Dict],
                           num_words: int) -> float:
        """Compute average Normalized PMI coherence across topics"""
        binary = (doc_term_matrix > 0).astype(int)
        num_docs = binary.shape[0]
        if num_docs < 2:
            return 0.0

        df = np.asarray(binary.sum(axis=0)).flatten()
        cooccur = (binary.T @ binary).toarray()

        topic_scores = []
        for topic in topics:
            idxs = [word_to_idx[w] for w in (t['word'] for t in topic['words'])
                    if w in word_to_idx]
            if len(idxs) < 2:
                continue

            pair_scores = []
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    df_ij = cooccur[i, j]
                    if df_ij == 0:
                        continue
                    pmi = math.log((df_ij * num_docs) / (df[i] * df[j]))
                    npmi = pmi / -math.log(df_ij / num_docs)
                    pair_scores.append(npmi)

            if pair_scores:
                topic_scores.append(sum(pair_scores) / len(pair_scores))

        if not topic_scores:
            return 0.0

        return round(sum(topic_scores) / len(topic_scores), 4)

    def _generate_topic_label(self, words: List[Dict]) -> str:
        """Generate a human-readable label for a topic"""
        # Use top 3 words
        top_words = [w['word'] for w in words[:3]]
        return " | ".join(top_words)

    def _get_document_topics(self, lda_model, doc_term_matrix, documents) -> List[Dict]:
        """Get topic distribution for each document"""
        doc_dist = lda_model.transform(doc_term_matrix)

        doc_topics = []
        for idx, dist in enumerate(doc_dist):
            # Sort by probability
            topic_dist = sorted(enumerate(dist), key=lambda x: x[1], reverse=True)
            doc_topics.append({
                "document_id": idx,
                "preview": documents[idx][:100] + "..." if len(documents[idx]) > 100 else documents[idx],
                "topics": [
                    {
                        "topic_id": topic_id,
                        "probability": round(float(prob), 4)
                    }
                    for topic_id, prob in topic_dist
                ]
            })

        return doc_topics


# Helper to tokenize and filter text
def word_tokenize_text(text: str, stop_words, punctuation) -> List[str]:
    """Tokenize, lowercase, and filter stopwords/punctuation"""
    # Simple whitespace + strip punctuation based tokenization (avoids nltk punkt dependency for use here)
    import re
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    tokens = [
        token for token in tokens
        if token not in stop_words
        and token not in punctuation
        and len(token) > 2
    ]
    return tokens


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
    print(f"Vocabulary Size: {result['vocabulary_size']}")
    print(f"\nTopics:")

    for topic in result['topics']:
        print(f"\nTopic {topic['topic_id']}: {topic['label']}")
        print("Words:")
        for word in topic['words']:
            print(f"  - {word['word']}: {word['probability']:.4f}")
