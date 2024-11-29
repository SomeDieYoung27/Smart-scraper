import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import requests
from io import BytesIO
import logging
from tqdm import tqdm
import os
from typing import List, Dict, Union
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorProcessor:
    def __init__(self, cache_dir: str = 'cache'):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-Chat")
        self.image_model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-Chat")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_model.to(self.device)

    def get_image_embedding(self, image_url: str) -> np.ndarray:
        """Generate embedding for an image using Qwen2-VL"""
        try:
            # Check cache first
            cache_key = str(hash(image_url))
            cache_path = self.cache_dir / f"img_{cache_key}.npy"
            
            if cache_path.exists():
                return np.load(cache_path)
            
            # Download and process image
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Process image with Qwen2-VL
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.image_model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy().mean(axis=1)
            
            # Cache the result
            np.save(cache_path, embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return np.zeros((1, 768))  # Return zero vector on error

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using SentenceTransformer"""
        try:
            # Check cache
            cache_key = str(hash(text))
            cache_path = self.cache_dir / f"text_{cache_key}.npy"
            
            if cache_path.exists():
                return np.load(cache_path)
            
            embedding = self.text_model.encode([text])[0]
            
            # Cache the result
            np.save(cache_path, embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return np.zeros(384)  # Return zero vector on error

    def process_course_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process course data to generate embeddings"""
        logger.info("Processing course data...")
        
        # Initialize lists for embeddings
        text_embeddings = []
        image_embeddings = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Generate text embedding from title and description
            text = f"{row['title']} {row['description']}"
            text_emb = self.get_text_embedding(text)
            text_embeddings.append(text_emb)
            
            # Generate image embedding if available
            if 'image_url' in row and pd.notna(row['image_url']):
                img_emb = self.get_image_embedding(row['image_url'])
                image_embeddings.append(img_emb)
            else:
                image_embeddings.append(np.zeros((1, 768)))
        
        # Add embeddings to dataframe
        df['text_embedding'] = text_embeddings
        df['image_embedding'] = image_embeddings
        
        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: str = 'output/courses_with_embeddings.pkl'):
        """Save processed dataframe with embeddings"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_pickle(output_path)
        logger.info(f"Processed data saved to {output_path}")

def main():
    # Load scraped data
    df = pd.read_csv('output/analytics_vidhya_courses.csv')
    
    # Process data
    processor = VectorProcessor()
    processed_df = processor.process_course_data(df)
    
    # Save processed data
    processor.save_processed_data(processed_df)

if __name__ == "__main__":
    main()
