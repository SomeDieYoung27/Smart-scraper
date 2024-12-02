import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import requests
from io import BytesIO
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import json

logging.basicConfig(level=logging.INFO)
logger = Logging.getLogger(__name__)


@dataclass
class SearchResult:
    title:str
    description:str
    rating:str
    link:str
    duration:str
    level:str
    relevance_score : float
    image_url : Optional[str] = None

class SmartSearchEngine:
    def __init__(self,data_path:str ="output/courses_with_embeddings.pkl"):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_processer = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-Chat")
        self.image_model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-Chat")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_model.to(self.device)
        

        #Loading course data
        self.df = pd.read_pickle(data_path)

        #Initialize FAISS indixes
        self.__init__faiss_indexes()

        #Cache for query embeddings
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

    def init_faiss_indexes(self):
        #Initialize FAISS indices for fast similarity search
        #Text embedding index
        text_dim = len(self.df['text_embedding'].iloc[0])
        self.text_index = faiss.IndexFlatIP(text_dim)
        text_vectors = np.stack(self.df['text_embedding'].values)
        self.text_index.add(text_vectors.astype('float32'))

        #Image embedding index
        image_dim = self.df['image_embedding'].iloc[0].shape[1]
        self.image_index = faiss.IndexFlatIp(image_dim)
        image_vectors = np.vstack(self.df['image_embedding'].values)
        self.image_index.add(image_vectors.astype('float32'))

        



