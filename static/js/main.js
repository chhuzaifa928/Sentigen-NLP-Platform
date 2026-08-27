// SentiGen - Frontend JavaScript
// Handles all user interactions and API calls

const API_BASE = '';

// Modal Management
const modals = {
    sentiment: document.getElementById('sentimentModal'),
    ner: document.getElementById('nerModal'),
    summarize: document.getElementById('summarizeModal'),
    generate: document.getElementById('generateModal'),
    topics: document.getElementById('topicsModal')
};

function openModal(modalId) {
    const modal = modals[modalId];
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal(modalId) {
    const modal = modals[modalId];
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = 'auto';

        // Clear results
        const results = modal.querySelector('.results');
        if (results) {
            results.classList.remove('active');
            results.innerHTML = '';
        }
    }
}

// Close modal when clicking outside
Object.values(modals).forEach(modal => {
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                const modalId = modal.id.replace('Modal', '');
                closeModal(modalId);
            }
        });
    }
});

// Sentiment Analysis
async function analyzeSentiment() {
    const text = document.getElementById('sentimentText').value.trim();
    const loading = document.getElementById('sentimentLoading');
    const results = document.getElementById('sentimentResults');
    const submitBtn = document.querySelector('#sentimentModal .btn-primary');

    if (!text) {
        alert('Please enter some text to analyze');
        return;
    }

    try {
        submitBtn.disabled = true;
        loading.classList.add('active');
        results.classList.remove('active');

        const response = await fetch(`${API_BASE}/api/sentiment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displaySentimentResults(data);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        submitBtn.disabled = false;
    }
}

function displaySentimentResults(data) {
    const results = document.getElementById('sentimentResults');

    const html = `
        <div class="result-section">
            <h3>Overall Sentiment</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${data.sentiment.label}</div>
                    <div class="stat-label">Sentiment</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(data.sentiment.confidence * 100).toFixed(1)}%</div>
                    <div class="stat-label">Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.primary_emotion}</div>
                    <div class="stat-label">Primary Emotion</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.polarity_score.toFixed(2)}</div>
                    <div class="stat-label">Polarity Score</div>
                </div>
            </div>
        </div>
        
        <div class="result-section">
            <h3>Emotion Analysis</h3>
            <ul class="emotion-list">
                ${data.emotions.map(emotion => `
                    <li class="emotion-item">
                        <div class="emotion-header">
                            <span>${emotion.emotion}</span>
                            <span>${(emotion.score * 100).toFixed(1)}%</span>
                        </div>
                        <div class="emotion-bar">
                            <div class="emotion-fill" style="width: ${emotion.score * 100}%"></div>
                        </div>
                    </li>
                `).join('')}
            </ul>
        </div>
        
        <div class="result-section">
            <h3>Analysis Summary</h3>
            <p class="text-secondary">${data.analysis}</p>
        </div>
    `;

    results.innerHTML = html;
    results.classList.add('active');
}

// Named Entity Recognition
async function extractEntities() {
    const text = document.getElementById('nerText').value.trim();
    const loading = document.getElementById('nerLoading');
    const results = document.getElementById('nerResults');
    const submitBtn = document.querySelector('#nerModal .btn-primary');

    if (!text) {
        alert('Please enter some text to analyze');
        return;
    }

    try {
        submitBtn.disabled = true;
        loading.classList.add('active');
        results.classList.remove('active');

        const response = await fetch(`${API_BASE}/api/ner`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayNERResults(data);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        submitBtn.disabled = false;
    }
}

function displayNERResults(data) {
    const results = document.getElementById('nerResults');

    const entityTypesList = Object.entries(data.entity_types)
        .map(([type, count]) => `<li>${type}: ${count}</li>`)
        .join('');

    const html = `
        <div class="result-section">
            <h3>Statistics</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${data.total_entities}</div>
                    <div class="stat-label">Total Entities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${Object.keys(data.entity_types).length}</div>
                    <div class="stat-label">Entity Types</div>
                </div>
            </div>
        </div>
        
        <div class="result-section">
            <h3>Highlighted Text</h3>
            <div class="entity-container">
                ${data.highlighted_html}
            </div>
        </div>
        
        <div class="result-section">
            <h3>Entity Types Found</h3>
            <ul class="text-secondary">
                ${entityTypesList}
            </ul>
        </div>
        
        ${data.statistics && data.statistics.most_common_entities ? `
            <div class="result-section">
                <h3>Most Common Entities</h3>
                <ul class="text-secondary">
                    ${data.statistics.most_common_entities.map(e =>
        `<li>${e.entity} (${e.count} times)</li>`
    ).join('')}
                </ul>
            </div>
        ` : ''}
    `;

    results.innerHTML = html;
    results.classList.add('active');
}

// Text Summarization
async function summarizeText() {
    const text = document.getElementById('summarizeText').value.trim();
    const method = document.getElementById('summarizeMethod').value;
    const loading = document.getElementById('summarizeLoading');
    const results = document.getElementById('summarizeResults');
    const submitBtn = document.querySelector('#summarizeModal .btn-primary');

    if (!text) {
        alert('Please enter some text to summarize');
        return;
    }

    try {
        submitBtn.disabled = true;
        loading.classList.add('active');
        results.classList.remove('active');

        const response = await fetch(`${API_BASE}/api/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                method,
                num_sentences: 3,
                max_length: 150
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displaySummarizeResults(data);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        submitBtn.disabled = false;
    }
}

function displaySummarizeResults(data) {
    const results = document.getElementById('summarizeResults');

    const html = `
        <div class="result-section">
            <h3>Summary</h3>
            <div style="padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 12px; border-left: 4px solid #667eea;">
                <p style="font-size: 1.05rem; line-height: 1.8;">${data.summary}</p>
            </div>
        </div>
        
        <div class="result-section">
            <h3>Statistics</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${data.method}</div>
                    <div class="stat-label">Method</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.original_length}</div>
                    <div class="stat-label">Original Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.summary_length}</div>
                    <div class="stat-label">Summary Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(data.compression_ratio * 100).toFixed(1)}%</div>
                    <div class="stat-label">Compression</div>
                </div>
            </div>
        </div>
    `;

    results.innerHTML = html;
    results.classList.add('active');
}

// Text Generation
async function generateText() {
    const prompt = document.getElementById('generatePrompt').value.trim();
    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxLength = parseInt(document.getElementById('maxLength').value);
    const loading = document.getElementById('generateLoading');
    const results = document.getElementById('generateResults');
    const submitBtn = document.querySelector('#generateModal .btn-primary');

    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    try {
        submitBtn.disabled = true;
        loading.classList.add('active');
        results.classList.remove('active');

        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                temperature,
                max_length: maxLength,
                num_sequences: 2
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayGenerateResults(data);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        submitBtn.disabled = false;
    }
}

function displayGenerateResults(data) {
    const results = document.getElementById('generateResults');

    const html = `
        <div class="result-section">
            <h3>Generated Text</h3>
            ${data.generated_texts.map((text, index) => `
                <div style="padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">Variation ${index + 1}</div>
                    <p style="font-size: 1.05rem; line-height: 1.8;">${text}</p>
                </div>
            `).join('')}
        </div>
        
        <div class="result-section">
            <h3>Generation Parameters</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${data.parameters.temperature}</div>
                    <div class="stat-label">Temperature</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.parameters.max_length}</div>
                    <div class="stat-label">Max Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.num_sequences}</div>
                    <div class="stat-label">Variations</div>
                </div>
            </div>
        </div>
    `;

    results.innerHTML = html;
    results.classList.add('active');
}

// Topic Modeling
let documentCount = 1;

function addDocument() {
    documentCount++;
    const container = document.getElementById('documentsContainer');
    const newDoc = document.createElement('div');
    newDoc.className = 'form-group';
    newDoc.innerHTML = `
        <label>Document ${documentCount}</label>
        <textarea class="topic-document" rows="3" placeholder="Enter document text..."></textarea>
    `;
    container.appendChild(newDoc);
}

async function discoverTopics() {
    const documentElements = document.querySelectorAll('.topic-document');
    const documents = Array.from(documentElements)
        .map(el => el.value.trim())
        .filter(text => text.length > 0);

    const numTopics = parseInt(document.getElementById('numTopics').value);
    const loading = document.getElementById('topicsLoading');
    const results = document.getElementById('topicsResults');
    const submitBtn = document.querySelector('#topicsModal .btn-primary');

    if (documents.length < 2) {
        alert('Please enter at least 2 documents');
        return;
    }

    try {
        submitBtn.disabled = true;
        loading.classList.add('active');
        results.classList.remove('active');

        const response = await fetch(`${API_BASE}/api/topics`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                documents,
                num_topics: numTopics,
                num_words: 10
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayTopicsResults(data);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        submitBtn.disabled = false;
    }
}

function displayTopicsResults(data) {
    const results = document.getElementById('topicsResults');

    const html = `
        <div class="result-section">
            <h3>Statistics</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${data.num_topics}</div>
                    <div class="stat-label">Topics Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.num_documents}</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.coherence_score.toFixed(3)}</div>
                    <div class="stat-label">Coherence Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.vocabulary_size}</div>
                    <div class="stat-label">Vocabulary Size</div>
                </div>
            </div>
        </div>
        
        <div class="result-section">
            <h3>Discovered Topics</h3>
            ${data.topics.map(topic => `
                <div style="padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
                    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
                        Topic ${topic.topic_id + 1}: ${topic.label}
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                        ${topic.words.slice(0, 8).map(word => `
                            <span style="background: rgba(102, 126, 234, 0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                                ${word.word} <span style="opacity: 0.7;">(${(word.probability * 100).toFixed(1)}%)</span>
                            </span>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
        </div>
    `;

    results.innerHTML = html;
    results.classList.add('active');
}

// Update temperature display
document.getElementById('temperature')?.addEventListener('input', (e) => {
    document.getElementById('temperatureValue').textContent = e.target.value;
});

// Update max length display
document.getElementById('maxLength')?.addEventListener('input', (e) => {
    document.getElementById('maxLengthValue').textContent = e.target.value;
});

// Update num topics display
document.getElementById('numTopics')?.addEventListener('input', (e) => {
    document.getElementById('numTopicsValue').textContent = e.target.value;
});
