# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Internet connection for downloading models

## Step-by-Step Installation

### 1. Clone or Download the Project

```bash
cd sentigen-nlp-platform
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn (web framework)
- Transformers (Hugging Face models)
- PyTorch (deep learning framework)
- spaCy (NLP library)
- NLTK (Natural Language Toolkit)
- Gensim (topic modeling)
- And other dependencies

**Note:** This may take 5-10 minutes depending on your internet speed.

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Download NLTK Data (Automatic)

The application will automatically download required NLTK data on first run.

## Running the Application

### Start the Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

## First Run

On the first run:
1. Models will be downloaded automatically (this may take a few minutes)
2. The application will cache models for faster subsequent runs
3. You may see download progress in the terminal

## Troubleshooting

### Issue: PyTorch Installation Fails

**Solution:** Install PyTorch separately:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: spaCy Model Not Found

**Solution:** Manually download the model:
```bash
python -m spacy download en_core_web_sm
```

### Issue: Out of Memory

**Solution:** 
- Close other applications
- Use smaller models (modify model names in code)
- Reduce batch sizes

### Issue: Port 8000 Already in Use

**Solution:** Use a different port:
```bash
uvicorn app:app --port 8001
```

## System Requirements

### Minimum:
- CPU: Dual-core processor
- RAM: 4GB
- Storage: 2GB free space

### Recommended:
- CPU: Quad-core processor or better
- RAM: 8GB or more
- Storage: 5GB free space
- GPU: Optional, but speeds up processing

## Model Download Sizes

- DistilBERT (sentiment): ~250MB
- RoBERTa (emotion): ~500MB
- GPT-2: ~500MB
- T5-small: ~250MB
- spaCy en_core_web_sm: ~15MB

**Total:** ~1.5GB of models will be downloaded

## Performance Tips

1. **First Request is Slow:** Models load on first use (lazy loading)
2. **GPU Acceleration:** Install CUDA-enabled PyTorch for faster processing
3. **Model Caching:** Models are cached after first download
4. **Memory Management:** Close unused applications when running

## Next Steps

After installation:
1. Read the [API Documentation](API.md)
2. Try the example requests
3. Explore the web interface
4. Customize for your needs

## Getting Help

If you encounter issues:
1. Check the terminal for error messages
2. Ensure all dependencies are installed
3. Verify Python version (3.8+)
4. Check available disk space and memory
