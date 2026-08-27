# API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "SentiGen NLP Platform is running"
}
```

---

### 2. Sentiment Analysis

**POST** `/api/sentiment`

Analyze sentiment and emotions in text.

**Request Body:**
```json
{
  "text": "I absolutely love this amazing product!"
}
```

**Response:**
```json
{
  "sentiment": {
    "label": "POSITIVE",
    "confidence": 0.9998
  },
  "emotions": [
    {
      "emotion": "joy",
      "score": 0.8542
    },
    {
      "emotion": "love",
      "score": 0.7231
    }
  ],
  "primary_emotion": "joy",
  "polarity_score": 0.8234,
  "analysis": "The text expresses a positive sentiment with joy as the dominant emotion..."
}
```

---

### 3. Named Entity Recognition

**POST** `/api/ner`

Extract named entities from text.

**Request Body:**
```json
{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."
}
```

**Response:**
```json
{
  "entities": [
    {
      "text": "Apple Inc.",
      "label": "ORG",
      "start": 0,
      "end": 10,
      "description": "Companies, agencies, institutions"
    },
    {
      "text": "Steve Jobs",
      "label": "PERSON",
      "start": 26,
      "end": 36,
      "description": "People, including fictional"
    }
  ],
  "total_entities": 4,
  "entity_types": {
    "ORG": 1,
    "PERSON": 1,
    "GPE": 2
  },
  "highlighted_html": "...",
  "statistics": {...}
}
```

---

### 4. Text Summarization

**POST** `/api/summarize`

Summarize text using extractive or abstractive methods.

**Request Body:**
```json
{
  "text": "Your long text here...",
  "method": "abstractive",
  "num_sentences": 3,
  "max_length": 150
}
```

**Parameters:**
- `text` (required): Text to summarize
- `method` (optional): "extractive" or "abstractive" (default: "abstractive")
- `num_sentences` (optional): Number of sentences for extractive (default: 3)
- `max_length` (optional): Max length for abstractive (default: 150)

**Response:**
```json
{
  "summary": "This is the generated summary...",
  "method": "abstractive",
  "original_length": 1250,
  "summary_length": 145,
  "compression_ratio": 0.116,
  "original_sentences": 15,
  "summary_sentences": 2
}
```

---

### 5. Text Generation

**POST** `/api/generate`

Generate text based on a prompt using GPT-2.

**Request Body:**
```json
{
  "prompt": "The future of artificial intelligence",
  "max_length": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "num_sequences": 2
}
```

**Parameters:**
- `prompt` (required): Starting text
- `max_length` (optional): Maximum length in tokens (default: 100)
- `temperature` (optional): 0.0-2.0, controls randomness (default: 0.7)
- `top_k` (optional): Limits vocabulary (default: 50)
- `top_p` (optional): Nucleus sampling (default: 0.9)
- `num_sequences` (optional): Number of variations (default: 1)

**Response:**
```json
{
  "prompt": "The future of artificial intelligence",
  "generated_texts": [
    "The future of artificial intelligence is bright and full of possibilities...",
    "The future of artificial intelligence will transform how we live and work..."
  ],
  "num_sequences": 2,
  "parameters": {
    "max_length": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
  },
  "model": "gpt2"
}
```

---

### 6. Topic Modeling

**POST** `/api/topics`

Discover topics in a collection of documents.

**Request Body:**
```json
{
  "documents": [
    "First document text...",
    "Second document text...",
    "Third document text..."
  ],
  "num_topics": 3,
  "num_words": 10
}
```

**Parameters:**
- `documents` (required): Array of at least 2 documents
- `num_topics` (optional): Number of topics to discover (default: 5)
- `num_words` (optional): Words per topic (default: 10)

**Response:**
```json
{
  "topics": [
    {
      "topic_id": 0,
      "words": [
        {
          "word": "machine",
          "probability": 0.0523
        },
        {
          "word": "learning",
          "probability": 0.0487
        }
      ],
      "label": "machine | learning | data"
    }
  ],
  "num_topics": 3,
  "num_documents": 5,
  "coherence_score": 0.4523,
  "document_topics": [...],
  "vocabulary_size": 156
}
```

---

### 7. Model Status

**GET** `/api/models/status`

Check which models are currently loaded.

**Response:**
```json
{
  "sentiment_analyzer": true,
  "ner_extractor": true,
  "text_summarizer": false,
  "text_generator": false,
  "topic_modeler": false
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message here"
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error

---

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider adding rate limiting middleware.

---

## Example Usage

### Python

```python
import requests

# Sentiment Analysis
response = requests.post(
    "http://localhost:8000/api/sentiment",
    json={"text": "This is amazing!"}
)
print(response.json())
```

### JavaScript

```javascript
fetch('http://localhost:8000/api/sentiment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'This is amazing!'})
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

---

## Performance Notes

1. **First Request:** Slower due to model loading
2. **Subsequent Requests:** Much faster (models cached)
3. **Concurrent Requests:** Supported
4. **Large Texts:** May take longer to process

---

## Best Practices

1. **Text Length:** Keep texts under 5000 characters for best performance
2. **Batch Processing:** For multiple texts, make separate requests
3. **Error Handling:** Always handle potential errors
4. **Timeouts:** Set appropriate timeouts for your client
