import re, nltk
from collections import Counter
from datetime import timedelta, datetime
from typing import List, Dict, Tuple

from hosts import load_clean_tweets, to_datetime, find_window, get_name_candidates
from nltk.sentiment import SentimentIntensityAnalyzer

def get_vader():
    try:
        # Check for the resource
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")

get_vader()
sia = SentimentIntensityAnalyzer()

humor_verbs =  re.compile(
    r"\b(lol|lmao|lmfao|rofl|haha+|hehe+|funny|hilarious|joke|jokes|joked|joking|roast|burn|zinger)\b",
    re.I
)

patterns = [
    re.compile(r"\bjoked about\s+([a-z0-9 \-']{3,80})", re.I),
    re.compile(r"\bjoke about\s+([a-z0-9 \-']{3,80})", re.I),
    re.compile(r"\bjokes about\s+([a-z0-9 \-']{3,80})", re.I),
    re.compile(r"\bthat\s+([a-z0-9 \-']{3,80})\s+joke\b", re.I),
    re.compile(r"\bthe\s+([a-z0-9 \-']{3,80})\s+joke\b", re.I),
]

stop_words = set("""
the a an of on in for to and or but with without about from at by into onto up down over under
""".split())

not_a_person = {w.lower() for w in {
    "Golden","Globes","Globe","Red","Carpet","Awards","Award",
    "Best","Worst","Present","Presenter","Arrivals","Arrival",
    "Hi","Your","Live","Tonight","Monologue","Opening"
}}

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

def sentiment_score(text_l: str) ->int:
    #figures out score
    
    score = sia.polarity_scores(text_l)["compound"]
    if score >=0.25:
        return 1
    elif score <= -0.4:
        return -1
    else: return 0

def trim_patterns(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw).strip(" -")
    tokens = [t for t in raw.split() if t.lower() not in stop_words]
    tokens = tokens[:6]

    while tokens and len(tokens[0]) <=2:
        tokens = tokens[1:]
    while tokens and len(tokens[-1]) <=2:
        tokens = tokens[:-1]
    
    return " ".join(tokens)

def find_themes(text: str) -> List[str]:
    themes = []
    text_l = text.lower()
    for pat in patterns:
        for m in pat.finditer(text_l):
            theme = trim_patterns(m.group(1))
            if theme and len(theme.split())>=1:
                themes.append(theme)
    return themes

def humor_window(cleaned_path: str, mins_after_start: int = 75) -> Tuple[datetime, datetime]:
    data = list(load_clean_tweets(cleaned_path))
    start, _ = find_window(data, window_minutes=40)
    return start, start + timedelta(minutes=mins_after_start)

def find_jokes(cleaned_path: str, top_k_people: int = 5, top_k_themes: int = 5) -> Dict[str, List[str]]:
    start, end = humor_window(cleaned_path)
    seen = set()
    people = Counter()
    themes = Counter()

    for tweet in load_clean_tweets(cleaned_path):
        text = (tweet.get("text", "") or "")
        if not text or text in seen:
            continue
        seen.add(text)

        ts = tweet.get("timestamp")
        if not ts:
            continue
        dt = to_datetime(ts)
        if not (start <= dt < end):
            continue

        text_l = text.lower()
        if not humor_verbs.search(text_l):
            continue

        score = sentiment_score(text_l)
        if score == 0:
            if not re.search(r"\b(joke|jokes|joked|joking)\b", text_l):
                continue
        
        names = [n for n in get_name_candidates(text) if likely_a_person(n)]
        for n in names:
            people[n]+=1
        
        for theme in find_themes(text):
            themes[theme]+=1

    funniest = [n for n,_ in people.most_common(top_k_people)]
    top_themes = [theme for theme, _ in themes.most_common(top_k_themes)]

    print("Funniest People:", funniest)
    print("Top Joke Themes:", top_themes)

    return {"funniest_people": funniest, "top_joke_themes": top_themes}
        

