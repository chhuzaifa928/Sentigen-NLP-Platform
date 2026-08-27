# Quick Start Guide

Get SentiGen up and running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- 4GB+ RAM
- Internet connection

## Installation (3 steps)

### 1. Install Dependencies

```bash
cd sentigen-nlp-platform
pip install -r requirements.txt
```

⏱️ This takes 5-10 minutes

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

⏱️ This takes 1-2 minutes

### 3. Run the Application

```bash
python app.py
```

⏱️ Server starts immediately!

## Access the Application

Open your browser and go to:
```
http://localhost:8000
```

## First Use

On first use, models will download automatically:
- ✅ DistilBERT (~250MB)
- ✅ RoBERTa (~500MB)
- ✅ GPT-2 (~500MB)
- ✅ T5-small (~250MB)

This happens only once! Models are cached for future use.

## Try It Out

### 1. Sentiment Analysis
- Click "Sentiment Analysis" card
- Enter: "I love this amazing product!"
- Click "Analyze Sentiment"
- See emotions and polarity scores

### 2. Named Entity Recognition
- Click "Named Entity Recognition" card
- Enter: "Apple Inc. was founded by Steve Jobs in California."
- Click "Extract Entities"
- See highlighted entities

### 3. Text Summarization
- Click "Text Summarization" card
- Paste a long article or text
- Choose method (Abstractive or Extractive)
- Click "Summarize"
- See concise summary

### 4. Text Generation
- Click "Text Generation" card
- Enter: "The future of AI is"
- Adjust temperature (creativity)
- Click "Generate Text"
- See AI-generated continuations

### 5. Topic Modeling
- Click "Topic Modeling" card
- Enter multiple documents (at least 2)
- Click "Add Document" for more
- Click "Discover Topics"
- See discovered topics

## Test the API

Run the test script:
```bash
python test_api.py
```

This tests all endpoints automatically!

## Troubleshooting

### Server won't start?
```bash
# Try a different port
uvicorn app:app --port 8001
```

### Models not downloading?
- Check internet connection
- Ensure enough disk space (2GB+)
- Wait patiently, downloads can take time

### Out of memory?
- Close other applications
- Restart your computer
- Try one feature at a time

## Next Steps

1. ✅ Read the full [README.md](README.md)
2. ✅ Check [API Documentation](API.md)
3. ✅ Explore [Installation Guide](INSTALLATION.md)

## Tips

- **First request is slow**: Models load on first use
- **Be patient**: AI processing takes time
- **Experiment**: Try different parameters
- **Have fun**: Explore all features!

## Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run application
python app.py

# Test API
python test_api.py

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

## That's It!

You're now ready to explore advanced NLP capabilities! 🚀

Enjoy using SentiGen! ✨
