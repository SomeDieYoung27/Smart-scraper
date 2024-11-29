import streamlit as st
import pandas as pd
from smart_search import SmartSearchEngine, format_results
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart Course Search",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .course-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        background-color: white;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
    }
    .metric {
        text-align: center;
        padding: 0.5rem;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .search-header {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_search_engine():
    """Load and cache the search engine"""
    return SmartSearchEngine()

def load_image_from_url(url):
    """Load image from URL for display"""
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Error loading image from {url}: {e}")
        return None

def display_course_card(result):
    """Display a single course result in a card format"""
    with st.container():
        st.markdown(f"""
        <div class="course-card">
            <h3>{result.title}</h3>
            <div class="metric-container">
                <div class="metric">
                    <strong>Level:</strong> {result.level}
                </div>
                <div class="metric">
                    <strong>Duration:</strong> {result.duration}
                </div>
                <div class="metric">
                    <strong>Rating:</strong> {result.rating}
                </div>
                <div class="metric">
                    <strong>Relevance:</strong> {result.relevance_score:.2f}
                </div>
            </div>
            <p>{result.description[:300]}...</p>
            <a href="{result.link}" target="_blank">View Course →</a>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize search engine
    search_engine = load_search_engine()
    
    # Sidebar
    st.sidebar.title("Search Options")
    
    # Search weights
    text_weight = st.sidebar.slider(
        "Text Search Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Weight given to text-based search results"
    )
    
    image_weight = st.sidebar.slider(
        "Image Search Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="Weight given to image-based search results"
    )
    
    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of courses to display"
    )
    
    # Main content
    st.title("🎓 Smart Course Search")
    st.markdown("""
    <div class="search-header">
        <p>Search for courses using text and optionally an image. 
        The search engine uses advanced AI models including Qwen2-VL for multimodal understanding.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter your search query (e.g., 'machine learning for beginners')"
        )
    
    with col2:
        image_url = st.text_input(
            "Image URL (Optional)",
            placeholder="Enter URL to a relevant image"
        )
    
    # Preview image if URL provided
    if image_url:
        image = load_image_from_url(image_url)
        if image:
            st.image(image, caption="Query Image", width=200)
    
    # Search button
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching for courses..."):
                # Perform search
                results = search_engine.search(
                    query=query,
                    image_query=image_url if image_url else None,
                    top_k=top_k,
                    text_weight=text_weight,
                    image_weight=image_weight
                )
                
                # Display results
                st.subheader(f"Found {len(results)} Courses")
                
                # Create relevance score visualization
                scores = [r.relevance_score for r in results]
                titles = [r.title for r in results]
                
                fig = px.bar(
                    x=scores,
                    y=titles,
                    orientation='h',
                    title='Course Relevance Scores',
                    labels={'x': 'Relevance Score', 'y': 'Course Title'}
                )
                st.plotly_chart(fig)
                
                # Display course cards
                for result in results:
                    display_course_card(result)
        else:
            st.warning("Please enter a search query")

if __name__ == "__main__":
    main()
