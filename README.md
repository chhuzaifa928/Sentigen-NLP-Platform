# SentiGen - Advanced NLP Analysis & Generation Platform

A comprehensive Natural Language Processing platform that combines state-of-the-art transformer models with classical NLP techniques to provide multi-dimensional text analysis and generation capabilities.

## 🌟 Features

### 1. **Multi-Aspect Sentiment Analysis**
- Emotion detection (joy, sadness, anger, fear, surprise)
- Tone analysis (positive, negative, neutral)
- Formality detection
- Confidence scores for each prediction

### 2. **Named Entity Recognition (NER)**
- Extract and classify entities: Person, Organization, Location, Date, etc.
- Visual entity highlighting
- Entity frequency analysis

### 3. **Text Summarization**
- Extractive summarization (key sentence extraction)
- Abstractive summarization using T5 transformer
- Customizable summary length

### 4. **Intelligent Text Generation**
- Context-aware text completion
- Multiple generation strategies (greedy, beam search, sampling)
- Temperature and top-k/top-p controls

### 5. **Topic Modeling**
- Latent Dirichlet Allocation (LDA)
- Discover hidden topics in documents
- Topic distribution visualization

### 6. **Text Similarity & Comparison**
- Semantic similarity using sentence transformers
- Document comparison
- Plagiarism detection capabilities

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **FastAPI** - Modern, fast web framework
- **Transformers** (Hugging Face) - BERT, GPT-2, T5, RoBERTa
- **spaCy** - Industrial-strength NLP
- **NLTK** - Natural Language Toolkit
- **scikit-learn** - Machine learning utilities
- **Gensim** - Topic modeling

### Frontend
- **HTML5/CSS3/JavaScript**
- **Modern UI/UX** with glassmorphism design
- **Chart.js** - Data visualization
- **Responsive design**

### Models Used
- `distilbert-base-uncased-finetuned-sst-2-english` - Sentiment analysis
- `cardiffnlp/twitter-roberta-base-emotion` - Emotion detection
- `gpt2` - Text generation
- `t5-small` - Abstractive summarization
- `en_core_web_sm` (spaCy) - NER and linguistic features
- `all-MiniLM-L6-v2` - Sentence embeddings

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd sentigen-nlp-platform
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run the application
```bash
python app.py
```

The application will be available at `http://localhost:8000`

## 🚀 Usage

### Web Interface
1. Navigate to `http://localhost:8000`
2. Select the NLP task you want to perform
3. Enter your text
4. View comprehensive analysis results

### API Endpoints

#### Sentiment Analysis
```bash
POST /api/sentiment
Content-Type: application/json

{
  "text": "I absolutely love this amazing product!"
}
```

#### Named Entity Recognition
```bash
POST /api/ner
Content-Type: application/json

{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."
}
```

#### Text Summarization
```bash
POST /api/summarize
Content-Type: application/json

{
  "text": "Your long text here...",
  "method": "abstractive",
  "max_length": 150
}
```

#### Text Generation
```bash
POST /api/generate
Content-Type: application/json

{
  "prompt": "The future of artificial intelligence",
  "max_length": 100,
  "temperature": 0.7
}
```

#### Topic Modeling
```bash
POST /api/topics
Content-Type: application/json

{
  "documents": ["doc1 text", "doc2 text", ...],
  "num_topics": 5
}
```

## 📊 Project Structure

```
sentigen-nlp-platform/
│
├── app.py                      # FastAPI application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/
│   ├── sentiment_analyzer.py  # Sentiment analysis module
│   ├── ner_extractor.py       # Named entity recognition
│   ├── summarizer.py          # Text summarization
│   ├── text_generator.py      # Text generation
│   └── topic_modeler.py       # Topic modeling
│
├── static/
│   ├── css/
│   │   └── style.css          # Styling
│   ├── js/
│   │   └── main.js            # Frontend logic
│   └── images/
│       └── logo.png
│
└── templates/
    └── index.html             # Main web interface
```

## 🎯 Key Highlights for Portfolio

1. **Multiple NLP Techniques** - Demonstrates breadth of NLP knowledge
2. **State-of-the-Art Models** - Uses latest transformer architectures
3. **Production-Ready** - FastAPI backend with proper error handling
4. **Beautiful UI** - Modern, responsive design
5. **Well-Documented** - Comprehensive README and code comments
6. **Scalable Architecture** - Modular design for easy extension

## 🔬 Technical Deep Dive

### Sentiment Analysis Pipeline
- Multi-model ensemble approach
- Combines transformer-based and rule-based methods
- Provides confidence scores and detailed breakdowns

### NER Implementation
- Uses spaCy's pre-trained models
- Custom entity highlighting
- Statistical analysis of entity distributions

### Summarization Strategy
- Dual approach: extractive and abstractive
- Extractive: TF-IDF and TextRank algorithms
- Abstractive: Fine-tuned T5 model

### Text Generation
- GPT-2 based generation
- Multiple decoding strategies
- Controllable generation parameters

## 📈 Future Enhancements

- [ ] Fine-tune models on custom datasets
- [ ] Add multilingual support
- [ ] Implement question-answering system
- [ ] Add text classification for multiple categories
- [ ] Integrate database for storing analysis history
- [ ] Add user authentication
- [ ] Deploy to cloud (AWS/GCP/Azure)

## 📝 License

MIT License

## 👤 Author

MUHAMMAD JABBAR ALAM

## 🙏 Acknowledgments

- Hugging Face for transformer models
- spaCy team for NLP tools
- FastAPI framework
