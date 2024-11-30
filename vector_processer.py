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
    def __init__(self,cache_dir:str="cache"):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-Chat")
        self.image_model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-Chat")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
