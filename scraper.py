import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import os
from PIL import Image
from io import BytesIO
import logging
from tqdm.asyncio import tqdm
import time
from typing import List,Dict,Optional
import numpy as np

logging.basicConfig(level=Logging.INFO)
logger = Logging.getLogger(__name__)

class ModelCourseScrapper:
    def __init__(self,base_url:str = "'https://courses.analyticsvidhya.com/collections/courses?page='"):
        self.base_url = base_url
        self.session = None
        self.courses_data : List[Dict] = []

    async def init_session(self):
        self.session = aiohttp.ClientSession()


    async def close_session(self):
        if self.session:
            await self.session.close()

    async def fetch_page(self,url:str,retries:int = 3):
        for attempt in range(retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error fetching page {url} : {e}")
                if attempt == retries-1:
                   return None;
                await asyncio.sleep(1)
        return None


    async def process_course_page(self,course_url:str) -> Dict:
        html = await self.fetch_page(course_url)
        if not html:
            return {}
        
        soup = BeautifulSoup(html,'html.parser')
        course_data = {}

        try:
           icon_par = soup.find('ul',class_="text_icon_list")
           values = icon_par.find_all("h4") if icon_par else []

           course_data = {
               'duration' : values[0].text if len(values > 0) else "NA",
               'rating' :  values[1].text if len(values > 1) else "NA",
               "level" : values[2].text if len(values>2) else "Beginner",
           }
           #Extract rich description
           desc_par = soup.find('section',class_="rich-text")
           course_data['description'] = desc_par.find('article').text.strip() if desc_par else "NA"

           #Extract source image if avalaible
           img_tag = soup.find('img',class_='course-image')
           if img_tag and img_tag.get('src'):
               course_data['image_url'] = img_tag.get('src')

        except Exception as e:
            logger.error(f"Error processing course page {course_url} : {e}")

        return course_data
    
    async def scrape_course(self,max_pages:int = 8):
        await self.init_session()

        try:
            for page in tqdm(range(1,max_pages+1),desc="Scraping pages"):
                page_url = f"{self.base_url}{page}"
                html = await self.fetch_page(page_url)

                if not html:
                    continue

                soup = BeautifulSoup(html,'html.parser')
                courses = soup.find_all('a',class_="course-card")

                for course in courses:
                    try:
                        title = course.find('h3').text.strip()
                        link = f"https://courses.analyticsvidhya.com{course['href']}"
                        
                        #Process individual course page
                        course_details = await self.process_course_page(link)

                        self.courses_data.append({
                            'title' : title,
                            'link' : link,
                            **course_details
                        })
                    except Exception as e:
                        logger.error(f"Error processing course : {e}")

                await asyncio.sleep(1)

        finally :
            await self.close_session()

    def save_to_csv(self,output_dir:str = "output"):
        "Saved scraped data to CSV"
        os.makedirs(output_dir,exist_ok=True)
        df = pd.DataFrame(self.courses_data)
        output_file = os.path.join(output_dir,"analytics_vidhya_courses.csv")
        df.to_csv(output_file,index=False)
        logger.info(f"Data saved to {output_file}")

    async def main():
        scraper = ModelCourseScrapper()
        await scraper.scrape_course()
        scraper.save_to_csv()

    if __name__ == "__main__":
        asyncio.run(main())








        
        
   
            