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

data_bk = reviews_all(
    'kr.co.burgerkinghybrid',
    lang='ko',
    country='kr',
    sort=Sort.NEWEST
)

data_mcd = reviews_all(
    'com.mcdonalds.mobileapp',
    lang='ko',
    country='kr',
    sort=Sort.NEWEST
)

data_moms = reviews_all(
    'kr.co.momstouch.moms',
    lang='ko',
    country='kr',
    sort=Sort.NEWEST
)

df_bk = pd.DataFrame(data_bk)
df_mcd = pd.DataFrame(data_mcd)
df_moms = pd.DataFrame(data_moms)

df_bk = filter_2years_ago(df_bk)
df_mcd = filter_2years_ago(df_mcd)
df_moms = filter_2years_ago(df_moms)

df_bk['app_name'] = 'Burger King'
df_mcd['app_name'] = 'McDonald\'s'
df_moms['app_name'] = 'Mom\'s Touch'

df = pd.concat([df_bk, df_mcd, df_moms], ignore_index=True)
df = df[['content', 'score','app_name']]

