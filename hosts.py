import json, re
from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

host_verbs = re.compile(
    r"\b(hosts?|hosting|hosted|your hosts|our hosts|please welcome|opening monologue)\b",
    re.IGNORECASE,
)

all_names = r"[A-Z][a-z']+"
name_chunk = re.compile(rf"\b({all_names}\s+{all_names}(?:\s+{all_names})?)\b")
name_pair  = re.compile(rf"\b({all_names}\s+{all_names})\s+(?:&|and)\s+({all_names}\s+{all_names})\b")



def load_clean_tweets(path: str) -> Iterable[dict]: 
    #want it to be a generator so we dont just return all tweets in a list
    #loads tweets - yields a dict
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def to_datetime(timestamp: str) -> datetime:
    #so we can add time/compare times
    return datetime.fromisoformat(timestamp) 

def find_window(tweets: Iterable[dict], window_minutes: int = 40):
    #want to return a window of estimate of when the hosting ceremony is
    per_min = Counter()
    first = None
    last = None

    #first pass - build up per_min
    for t in tweets:
        text = t.get("text", "")
        hashtags = t.get("hashtags", []) or []
        if not text:
            continue
        host_likely = bool(host_verbs.search(text)) or any("host" in h.lower() for h in hashtags) or "opening monologue" in text 
        if not host_likely:
            continue
        #get the time of the tweet
        dt = to_datetime(t["timestamp"])
        current_min = dt.replace(second = 0, microsecond = 0)
        #build counter holding num relationary tweets in that specific time
        per_min[current_min] +=1

        if first is None or dt<first:
            first = dt
        if last is None or dt> last:
            last = dt

    #now get the best window - most amount of host related tweets
    if not per_min:
        #in case per_min wasnt filled out, start at first timestamp if it exists
        all_times =[to_datetime(t["timestamp"]) for t in tweets]
        if all_times:
            start = min(all_times)
        else:
            start = datetime.now()
        return (start, start+timedelta(minutes=window_minutes))
    
    key_mins = sorted(per_min.keys())
    best_total, best_start = -1, None

    for start_min in key_mins:
        end_min = start_min + timedelta(minutes = window_minutes)
        total = 0
        cur = start_min
            
        while cur<end_min:
            total += per_min.get(cur, 0)
            cur += timedelta(minutes=1)

        #we want the window with the most amount of tweets relating to hosts
        if total > best_total:
            best_total = total
            best_start = start_min

    return(best_start, best_start+timedelta(minutes=window_minutes)) 

def clean_name(s:str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    names = s.split()
    if not (2 <= len(names) <= 3):  # prefer at least first+last
        return ""
    if not all(re.match(r"^[A-Z][a-z']+$", n) for n in names):
        return ""
    return " ".join(names)

def get_name_candidates(text: str) -> List[str]:
    #get first names and/or pairs
    candidates: List[str] = []

    # capture explicit pairs (x and y)
    for a, b in name_pair.findall(text):
        name_a = clean_name(a)
        name_b = clean_name(b)
        if name_a:
            candidates.append(name_a)
        if name_b:
            candidates.append(name_b)

    for n in name_chunk.finditer(text):
        name = clean_name(n.group(1))
        if name:
            candidates.append(name)

    #no repeats
    out, seen = [], set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    return out

def finalize_hosts(top_hosts: List[Tuple[str, int]]) -> List[str]:
    if not top_hosts:
        return []
    
    if len(top_hosts)==1:
        return [top_hosts[0][0]]
    
    top_name, top_score = top_hosts[0]
    second_name, second_score = top_hosts[1]
    if second_score == 0 or top_score>= 2*second_score:
        #if first is more than double second then likely only host
        return [top_name]
    
    return [top_name, second_name]

def find_hosts(cleaned_path: str, drop_retweets: bool=True, window_minutes: int=40) -> List[str]:

    #takes in tweets - returns host name
    data = list(load_clean_tweets(cleaned_path))
    start, end = find_window(data, window_minutes=window_minutes)

    scores = Counter()
    for tweet in data:
        if drop_retweets and tweet.get("is_retweet"):
            continue #in case we dont want to count retweets

        text = tweet.get("text", "")
        if not text:
            continue
        dt = to_datetime(tweet["timestamp"])
        if not (start <=dt <end):
            continue #dont want tweets not in our window
        
        hashtags = tweet.get("hashtags", []) or []
        has_host = bool(host_verbs.search(text)) or any("host" in (h or "").lower() for h in hashtags)
        if not has_host:
            continue #want tweets that are host related

        base = 2 if ("opening monologue" in text.lower() or "please welcome" in text.lower()) else 1

        for name in get_name_candidates(text):
            scores[name] += base
        
    if not scores:
        return []
    

    #print(scores.most_common(5))
    hosts = finalize_hosts(scores.most_common(2))
  
    return hosts