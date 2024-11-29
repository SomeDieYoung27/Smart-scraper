# Smart Course Search Engine

A modern, AI-powered course search engine that uses multimodal RAG (Retrieval-Augmented Generation) techniques to provide highly relevant course recommendations. The system combines text and image understanding using state-of-the-art models including Qwen2-VL for visual understanding and Sentence Transformers for text processing.

## Features

- **Multimodal Search**: Search using both text queries and relevant images
- **Advanced AI Models**:
  - Qwen2-VL for visual understanding
  - SentenceTransformer for text embeddings
  - FAISS for fast similarity search
- **Modern Web Interface**:
  - Built with Streamlit
  - Interactive visualizations using Plotly
  - Responsive design
  - Real-time search results
- **Smart Features**:
  - Cached embeddings for faster responses
  - Weighted multimodal similarity scoring
  - Course recommendations based on content similarity
  - Detailed course information including duration, level, and ratings

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- 8GB+ RAM recommended
- Internet connection for model downloads and image processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd smart_course_search
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Optional: For GPU support, install additional dependencies:
```bash
pip install cupy-cuda11x faiss-gpu
```

## Usage

1. First, scrape the course data:
```bash
python scraper.py
```

2. Generate embeddings:
```bash
python vector_processor.py
```

3. Run the web interface:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
smart_course_search/
├── app.py              # Streamlit web interface
├── scraper.py          # Async course data scraper
├── vector_processor.py # Embedding generation
├── smart_search.py     # Search engine core
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
├── cache/             # Cached embeddings
└── output/            # Processed data files
```

## How It Works

1. **Data Collection**: The scraper asynchronously collects course data including titles, descriptions, ratings, and images.

2. **Vector Processing**:
   - Text content is processed using SentenceTransformer
   - Images are processed using Qwen2-VL
   - Embeddings are cached for performance

3. **Search Process**:
   - User inputs text query and optional image
   - System generates query embeddings
   - FAISS performs similarity search
   - Results are ranked using weighted scoring
   - Top results are displayed with relevance scores

4. **User Interface**:
   - Interactive search controls
   - Adjustable search weights
   - Visual result presentation
   - Course details in card format

## Advanced Features

### Multimodal Understanding
The system uses Qwen2-VL, a state-of-the-art vision-language model, to understand both textual and visual content. This enables more nuanced search capabilities where visual elements can enhance the search accuracy.

### Smart Caching
The system implements intelligent caching of embeddings to improve performance:
- Text embeddings are cached based on content hash
- Image embeddings are cached based on URL hash
- Cache is automatically managed to prevent excessive storage use

### Weighted Search
Users can adjust the importance of text vs. image similarity in the search results, allowing for more personalized search experiences.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
