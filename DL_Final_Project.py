!pip install app_store_scraper
!pip install google-play-scraper

import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm
from app_store_scraper import AppStore
from google_play_scraper import  Sort, reviews_all
from datetime import datetime, timedelta

def filter_2years_ago(df):
  df['at'] = pd.to_datetime(df['at'])
  two_years_ago = datetime.now() - timedelta(days=2*365)
  return df[df['at'] >= two_years_ago]

