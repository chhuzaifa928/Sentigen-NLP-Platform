"""
Named Entity Recognition Module
Extracts and classifies named entities from text using spaCy
"""

import spacy
from typing import Dict, List, Tuple
from collections import Counter
import html
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NERExtractor:
    def __init__(self):
        """Initialize NER model"""
        try:
            logger.info("Loading spaCy NER model...")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("NER extractor initialized successfully!")
            
        except OSError:
            logger.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing entities, statistics, and highlighted text
        """
        try:
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_)
                })
            
            # Entity statistics
            entity_counts = Counter([ent['label'] for ent in entities])
            
            # Generate highlighted HTML
            highlighted_text = self._generate_highlighted_html(text, entities)
            
            return {
                "entities": entities,
                "total_entities": len(entities),
                "entity_types": dict(entity_counts),
                "highlighted_html": highlighted_text,
                "statistics": self._generate_statistics(entities, entity_counts)
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {
                "error": str(e),
                "entities": [],
                "total_entities": 0,
                "entity_types": {},
                "highlighted_html": text,
                "statistics": {}
            }
    
    def _generate_highlighted_html(self, text: str, entities: List[Dict]) -> str:
        """Generate HTML with highlighted entities"""
        if not entities:
            return text
        
        # Color mapping for different entity types
        colors = {
            "PERSON": "#FF6B6B",
            "ORG": "#4ECDC4",
            "GPE": "#45B7D1",
            "DATE": "#FFA07A",
            "TIME": "#98D8C8",
            "MONEY": "#F7DC6F",
            "PERCENT": "#BB8FCE",
            "PRODUCT": "#85C1E2",
            "EVENT": "#F8B739",
            "LOC": "#52B788",
            "WORK_OF_ART": "#E07A5F",
            "LAW": "#81B29A",
            "LANGUAGE": "#F2CC8F"
        }
        
        # Sort entities by start position (ascending for sequential building)
        sorted_entities = sorted(entities, key=lambda x: x['start'])

        result_parts = []
        pos = 0
        for ent in sorted_entities:
            result_parts.append(html.escape(text[pos:ent['start']]))
            color = colors.get(ent['label'], "#95A5A6")
            result_parts.append(
                f'<span class="entity" style="background-color: {color}; '
                f'padding: 2px 6px; border-radius: 4px; color: white; '
                f'font-weight: 500; margin: 0 2px;" '
                f'title="{html.escape(ent["label"])}: {html.escape(ent["description"])}">'
                f'{html.escape(ent["text"])}'
                f'<span class="entity-label" style="font-size: 0.7em; '
                f'margin-left: 4px; opacity: 0.9;">{html.escape(ent["label"])}</span>'
                f'</span>'
            )
            pos = ent['end']
        result_parts.append(html.escape(text[pos:]))

        return ''.join(result_parts)
    
    def _generate_statistics(self, entities: List[Dict], entity_counts: Counter) -> Dict:
        """Generate entity statistics"""
        if not entities:
            return {}
        
        # Most common entities
        entity_texts = [ent['text'] for ent in entities]
        most_common = Counter(entity_texts).most_common(5)
        
        return {
            "most_common_entities": [
                {"entity": text, "count": count}
                for text, count in most_common
            ],
            "entity_type_distribution": [
                {"type": ent_type, "count": count, "percentage": round(count / len(entities) * 100, 2)}
                for ent_type, count in entity_counts.most_common()
            ]
        }
    
    def extract_relationships(self, text: str) -> List[Dict]:
        """Extract simple relationships between entities"""
        doc = self.nlp(text)
        relationships = []
        
        for sent in doc.sents:
            sent_ents = list(sent.ents)
            if len(sent_ents) >= 2:
                # Find verb between entities
                for i in range(len(sent_ents) - 1):
                    ent1 = sent_ents[i]
                    ent2 = sent_ents[i + 1]
                    
                    # Find tokens between entities
                    between_tokens = doc[ent1.end:ent2.start]
                    verbs = [token.text for token in between_tokens if token.pos_ == "VERB"]
                    
                    if verbs:
                        relationships.append({
                            "subject": ent1.text,
                            "subject_type": ent1.label_,
                            "relation": " ".join(verbs),
                            "object": ent2.text,
                            "object_type": ent2.label_
                        })
        
        return relationships


# Test function
if __name__ == "__main__":
    extractor = NERExtractor()
    
    test_text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    The company is headquartered in Cupertino, California. In 2023, Apple reported revenue 
    of $394.3 billion. Tim Cook became CEO in August 2011, succeeding Steve Jobs.
    """
    
    result = extractor.extract_entities(test_text)
    
    print(f"Total Entities: {result['total_entities']}")
    print(f"\nEntity Types: {result['entity_types']}")
    print(f"\nEntities:")
    for ent in result['entities']:
        print(f"  - {ent['text']} ({ent['label']}): {ent['description']}")
    
    print(f"\nStatistics:")
    print(result['statistics'])
