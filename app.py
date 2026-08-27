"""
SentiGen - Advanced NLP Platform
FastAPI Backend Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import os

# Import NLP models
from models import (
    SentimentAnalyzer,
    NERExtractor,
    TextSummarizer,
    TextGenerator,
    TopicModeler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SentiGen NLP Platform",
    description="Advanced Natural Language Processing API",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response Models
class TextInput(BaseModel):
    text: str

class SummarizeInput(BaseModel):
    text: str
    method: str = "abstractive"
    num_sentences: int = 3
    max_length: int = 150

class GenerateInput(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    num_sequences: int = 1

class TopicInput(BaseModel):
    documents: List[str]
    num_topics: int = 5
    num_words: int = 10

# Initialize NLP models (lazy loading)
sentiment_analyzer = None
ner_extractor = None
text_summarizer = None
text_generator = None
topic_modeler = None


def get_sentiment_analyzer():
    """Lazy load sentiment analyzer"""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        logger.info("Initializing Sentiment Analyzer...")
        sentiment_analyzer = SentimentAnalyzer()
    return sentiment_analyzer


def get_ner_extractor():
    """Lazy load NER extractor"""
    global ner_extractor
    if ner_extractor is None:
        logger.info("Initializing NER Extractor...")
        ner_extractor = NERExtractor()
    return ner_extractor


def get_text_summarizer():
    """Lazy load text summarizer"""
    global text_summarizer
    if text_summarizer is None:
        logger.info("Initializing Text Summarizer...")
        text_summarizer = TextSummarizer()
    return text_summarizer


def get_text_generator():
    """Lazy load text generator"""
    global text_generator
    if text_generator is None:
        logger.info("Initializing Text Generator...")
        text_generator = TextGenerator()
    return text_generator


def get_topic_modeler():
    """Lazy load topic modeler"""
    global topic_modeler
    if topic_modeler is None:
        logger.info("Initializing Topic Modeler...")
        topic_modeler = TopicModeler()
    return topic_modeler


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Welcome to SentiGen NLP Platform</h1>", status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "SentiGen NLP Platform is running"}


@app.post("/api/sentiment")
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of text
    
    Returns multi-aspect sentiment analysis including emotions
    """
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        analyzer = get_sentiment_analyzer()
        result = analyzer.analyze_sentiment(input_data.text)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ner")
async def extract_entities(input_data: TextInput):
    """
    Extract named entities from text
    
    Returns entities with types, positions, and statistics
    """
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        extractor = get_ner_extractor()
        result = extractor.extract_entities(input_data.text)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in NER: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
async def summarize_text(input_data: SummarizeInput):
    """
    Summarize text using extractive or abstractive methods
    
    Methods:
    - extractive: Select key sentences from original text
    - abstractive: Generate new summary using T5
    """
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if input_data.method not in ["extractive", "abstractive"]:
            raise HTTPException(status_code=400, detail="Method must be 'extractive' or 'abstractive'")
        
        summarizer = get_text_summarizer()
        result = summarizer.summarize(
            input_data.text,
            method=input_data.method,
            num_sentences=input_data.num_sentences,
            max_length=input_data.max_length
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate_text(input_data: GenerateInput):
    """
    Generate text based on a prompt using GPT-2
    
    Parameters:
    - temperature: Controls randomness (0.0-2.0)
    - top_k: Limits vocabulary to top k tokens
    - top_p: Nucleus sampling threshold
    """
    try:
        if not input_data.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if not 0.0 <= input_data.temperature <= 2.0:
            raise HTTPException(status_code=400, detail="Temperature must be between 0.0 and 2.0")
        
        generator = get_text_generator()
        result = generator.generate(
            input_data.prompt,
            max_length=input_data.max_length,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            top_p=input_data.top_p,
            num_return_sequences=input_data.num_sequences
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/topics")
async def discover_topics(input_data: TopicInput):
    """
    Discover topics in a collection of documents using LDA
    
    Requires at least 2 documents
    """
    try:
        if len(input_data.documents) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 documents for topic modeling")
        
        if input_data.num_topics < 1:
            raise HTTPException(status_code=400, detail="Number of topics must be at least 1")
        
        modeler = get_topic_modeler()
        result = modeler.discover_topics(
            input_data.documents,
            num_topics=input_data.num_topics,
            num_words=input_data.num_words
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in topic modeling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/status")
async def models_status():
    """Get status of loaded models"""
    return {
        "sentiment_analyzer": sentiment_analyzer is not None,
        "ner_extractor": ner_extractor is not None,
        "text_summarizer": text_summarizer is not None,
        "text_generator": text_generator is not None,
        "topic_modeler": topic_modeler is not None
    }


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting SentiGen NLP Platform...")
    logger.info("Access the application at http://localhost:8000")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
