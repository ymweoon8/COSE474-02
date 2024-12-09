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
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re

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

# 리뷰의 긍정/부정 여부를 분류하기 위한 BERT 모델 로드
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 리뷰 데이터를 긍정/부정으로 분리하는 함수
def classify_reviews(df):
    df['sentiment'] = df['content'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
    df['sentiment'] = df['sentiment'].map({
        '1 star': 'negative',
        '2 stars': 'negative',
        '3 stars': 'neutral',
        '4 stars': 'positive',
        '5 stars': 'positive'
    })
    return df

# 특정 이슈 관련 키워드 추출을 위한 패턴 매칭 함수
def extract_issue_keywords(reviews, patterns):
    issue_keywords = {}
    for review in reviews:
        for issue, pattern in patterns.items():
            if re.search(pattern, review):
                if issue not in issue_keywords:
                    issue_keywords[issue] = 0
                issue_keywords[issue] += 1
    sorted_issues = sorted(issue_keywords.items(), key=lambda x: x[1], reverse=True)
    return sorted_issues

issue_patterns = {
    '로그인 문제': r'로그인|계정|자동로그인|비밀번호|아이디',
    '주문 실패': r'주문|오더|장바구니',
    '쿠폰 문제': r'쿠폰|프로모션|할인',
    '결제 오류': r'결제|환불|결제오류',
    '앱 충돌': r'꺼짐|멈춤|충돌|팅김|중지',
    '위치 서비스': r'GPS|위치|지도|반경',
    '알림 문제': r'푸시|알림',
    '속도 문제': r'느림|느려|속도|로딩|딜레이|렉|멈춤',
    '인터페이스': r'UI|화면|디자인|사용법|이용 방법',
    '회원가입 문제': r'회원가입|가입',
    '배달 문제': r'배달|딜리버리|시간|식음',
    '앱 설치 문제': r'설치|업데이트',
    '데이터 문제': r'데이터|정보|저장|초기화',
    '품질 문제': r'품질|음식|재료|상태',
}
