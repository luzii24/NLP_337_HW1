import json, re, os, pathlib
from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple, Dict

from hosts import load_clean_tweets, to_datetime, find_window, get_name_candidates
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from icrawler.builtin import BingImageCrawler


def get_vader():
    try:
        # Check for the resource
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")

get_vader()
sia = SentimentIntensityAnalyzer()


red_carpet_verbs = re.compile(
    r"\b(red carpet|#redcarpet|#eredcarpet|manicam|mani cam|arrivals?)\b",
    re.IGNORECASE,
)
outfit_verbs = re.compile(
    r"\b(dress|gown|tux|suit|outfit|look|train|sequin|sequins|lace|neckline|hem|fit|tailor|styled?)\b",
    re.IGNORECASE,
)
not_a_person = {w.lower() for w in {
    "Golden","Globes","Globe","Red","Carpet","Awards","Award",
    "Best","Worst","Present","Presenter","Arrivals","Arrival",
    "Hi","Your","Live","Tonight","Monologue","Opening"
}}
best_re  = re.compile(r"\bbest[-\s]?dressed\b", re.I)
worst_re = re.compile(r"\bworst[-\s]?dressed\b", re.I)

#************ Helpers ***************

def likely_a_person(name:str) -> bool:
    parts = name.split()
    if not (2 <= len(parts) <=3):
        return False
    for p in parts:
        if not re.match(r"^[A-Z][a-z']+$", p):
            return False
        if p.lower() in not_a_person:
            return False
    return True

def sentiment_score(text: str) ->int:
    #figures out score
    
    score = sia.polarity_scores(text)["compound"]
    if score >=0.3:
        return 1
    elif score <= -0.3:
        return -1
    else: return 0

def ceremony_window(cleaned_path: str, minutes: int = 45) -> Tuple[datetime, datetime]:
    #use hosts.py to get best opening ceremony window
    data = list(load_clean_tweets(cleaned_path))
    start, end = find_window(data, window_minutes=minutes)
    return start, end

def redcarpet_window(cleaned_path: str, ceremony_start: datetime, max_prior_minutes: int =120) -> Tuple[datetime, datetime]:
    #the redcarpet will be before the host/opening ceremony
    #max_prior_minutes = the max time it could be before the hosting ceremony - lets say 2 hrs
    data = list(load_clean_tweets(cleaned_path))
    per_min = Counter()

    for tweet in data:
        text = (tweet.get("text", "") or "")
        if not text:
            continue
        timestamp = tweet.get("timestamp")
        if not timestamp:
            continue
        
        dt = to_datetime(timestamp)
        if dt>=ceremony_start:
            #only want ones before ceremony not after
            continue

        text_l = text.lower()
        if red_carpet_verbs.search(text_l) or outfit_verbs.search(text_l):
            minute = dt.replace(second=0, microsecond=0)
            per_min[minute] +=1

    if per_min:
        #holds the time and amount of relevant tweets at that minute
        minutes = sorted(m for m in per_min if ceremony_start - timedelta(minutes=max_prior_minutes) <= m < ceremony_start)
            #get the minutes between the earliest it could be and the latest (hosting)
        if minutes:
            best_total, best_start = -1, minutes[0]
            for start_min in minutes:
                end = start_min + timedelta(minutes=60) #one hour span
                total, current = 0, start_min
                while current < end:
                    total += per_min.get(current, 0)
                    current += timedelta(minutes=1)
                if total>best_total:
                    best_total, best_start = total, start_min
            
            return best_start, best_start+timedelta(minutes=60)
    
    return ceremony_start-timedelta(minutes=75), ceremony_start
    #in case we dont get ggood window - just look at hour before hosting

#******** Produce Images ************

def safe_dir(p:str) ->str:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

def download_bing(query: str, out_dir: str, max_num: int =1, min_size: tuple = (256, 256)) -> None:
    out_dir = safe_dir(out_dir)
    crawler = BingImageCrawler(
        storage={"root_dir": out_dir},
        downloader_threads = 2,
        parser_threads = 2
    )

    try:
        crawler.crawl(
            keyword=query,
            max_num = max_num,
            min_size = min_size,
            file_idx_offset = 0
        )
    except Exception as e:
        print(f"[img] '{query}' failed: {e}")

def download_looks(best: List[str], worst: List[str], year: str, per_person: int=1) -> None: 
    for bucket, names in (("best", best), ("worst", worst)):
        for name in names:
            q = f"{name} golden globes {year} red carpet"
            out = os.path.join("red_carpet", bucket, name.replace(" ", "_"))
            download_bing(q, out_dir=out, max_num=per_person)

    print("Images saved under red_carpet/best and red_carpet/worst")          

#********* Main Function ************


def find_best_worst(cleaned_path: str, year: str, top_k: int = 5) -> Dict[str, List[str]]:    #want best_dressed:...., worst_dressed:...
    ceremony_start, _ = ceremony_window(cleaned_path, minutes=45)
    rc_start, rc_end = redcarpet_window(cleaned_path, ceremony_start)

    pos_scores = Counter()
    neg_scores = Counter()
    tweets = load_clean_tweets(cleaned_path)

    for tweet in tweets:
        text = (tweet.get("text", "") or "")
        if not text:
            continue
        
        timestamp = tweet.get("timestamp")
        if not timestamp:
            continue
        
        dt = to_datetime(timestamp)
        if not (rc_start <= dt < rc_end):
            continue
        
        text_l = text.lower()
        if not (red_carpet_verbs.search(text_l) or outfit_verbs.search(text_l)):
            continue

        label = 0
        if best_re.search(text_l):
            label = 1
        elif worst_re.search(text_l):
            label = -1
        else:
            label = sentiment_score(text_l)
        if label ==0:
            continue

        names = [n for n in get_name_candidates(text) if likely_a_person(n)]
        if not names:
            continue

        if label >0:
            for n in names:
                pos_scores[n] +=1
        else:
            for n in names:
                neg_scores[n] += 1

    best = [n for n, _ in pos_scores.most_common(top_k)]
    worst = [n for n, _ in neg_scores.most_common(top_k)]

    download_looks([best[0]], [worst[0]], year=year, per_person=1)
    print("Best Dressed:", best)
    print("Worst Dressed:", worst)
    
    return {"best_dressed": best, "worst_dressed": worst}