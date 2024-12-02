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


    def get_query_embeddings(self,text_query:str,image_query:Optional[str] = None) -> tuple:
        #Generate embeddings for text and image queries
        text_emb = self.text_model.encode([text_query])[0]

        #Get image embeddings if you can(image embeddings if you can)
        if image_query:
            try:
                response = requests.get(image_query)
                image = Image.open(BytesIO(response.content)).convert*('RGB')
                inputs = self.image_processer(images=image,return_tensors="pt").to(self.device)


                with torch.no_grad():
                    image_features =self.image_model.get_image_features(**inputs)
                    image_emb = image_features.cpu().numpy().mean(axis=1)

            except Exception as e:
                logger.error(f"Error processing image query: {e}")
                image_emb = None
            
            else :
                image_emb = None

            return text_emb,image_emb

        def search(self,query:str,image_query:Optional[str]=None,top_k:int=5,text_weight:float=0.7,image_weight:float=0.3) -> List[SearchResult]:

             """
             Perform multimodal search using both text and image queries
        
              Args:
              query: Text search query
              image_query: Optional URL to an image for visual search
              top_k: Number of results to return
              text_weight: Weight given to text similarity (0-1)
              image_weight: Weight given to image similarity (0-1)
              """

      
             text_emb,image_emb = self.get_query_embeddings(query,image_query)

            #Get text similarities
             text_emb = text_emb.reshape(1,-1).astype('float32')
             text_scores,text_indices = self.text_index.search(text_emb,len(self.df))

            #Get image similarities if avaliable
             if image_emb is not None:
                image_scores,image_indices = self.image_index.search(image_emb.astype('float32'),len(self.df))

                #Combine the scores
                combined_scores = (text_weight * text_scores[0] + image_weight * image_scores[0]) 

             else : 
                 combined_scores = text_scores[0]

            #Sort by combined scores
             top_indices = np.argsort(combined_scores)[:top_k][::-1]

            #Preparing results
             results = []
             for idx in top_indices:
                course = self.df.iloc[idx]
                results.append(
                    SearchResult(
                        title = course['title'],
                        description=course['description'],
                        rating = course['rating'],
                        link = course['link'],
                        duration = course['duration'],
                        level=course['level'],
                        relevance_score=float(combined_scores[idx]),
                        image_url=course['image_url']
                    )
                )
             return results














