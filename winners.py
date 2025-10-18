import json, re
from collections import Counter, defaultdict
from typing import Dict, List, Iterable, Tuple

# patterns to look for
win_verbs = [
    r"\bwin\b",
    r"\bwon\b",
    r"\bwins\b",
    r"\bgoes to\b",
    r"\baward goes to\b",
    r"\btakes home\b"]
#matches exactly the word win, wins, goes to, award goes to, takes home
win_re = re.compile(
    "|".join(win_verbs), 
    flags=re.IGNORECASE)
#matches with tweets that match with the winning verbs

negation_re = re.compile(
    r"\b(should(?:\s+have)?\s+won|robbed|snubbed|did(?:n['’]t| not)\s+win|deserved to win)\b",
    flags= re.IGNORECASE)
#matches with cases we want to ignore - when they didnt win
    #should have won or should won

future_re = re.compile(
    r"\b(will|should)\s+win\b",
    flags=re.IGNORECASE)
#matches will win or should win - not decided yet so we should exclude

candidate_window = 120
candidate_span = re.compile(
    r"[a-z0-9&'().,:!\-\s]{2,}", 
    flags=re.IGNORECASE)
#Match tweets that are regular words/sentences
    #easier to look through/find winners out of

#helper functions
def load_clean_tweets(path: str) -> Iterable[dict]: 
    #want it to be a generator so we dont just return all tweets in a list
    #loads tweets - yields a dict
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def set_window(text:str, index:int, w:int = candidate_window ) -> str:
    #takes text and returns portion of it based on the window size
        #if Won is found in tweet then returns section around that
    low = max(0, index-w)
    high = min(len(text), index+w)
    return text[low:high]

def pattern_weight(text:str) -> int:
    #want to be able to rank candidates - give priority/strength
    s=text.lower()
    if "award goes to" in s or "goes to" in s: 
        return 3
    if "won" in s or "wins" in s: 
        return 3
    if "takes home" in s: 
        return 2
    return 1

def split_candidates(win_clause: str) ->List[str]:
    #Takes in clause and extracts possible winner names/titles
    quoted = re.findall(r"[\"“”']([^\"“”']{2,})[\"“”']", win_clause)
        #Look at things between quotes - used for names/titles
    parts: List[str]=[]
    if quoted:
        parts.extend(quoted)
    
    for seg in re.split(r"\band\b|,|/|&", win_clause):
        #want to break apart at and,comma,& etc
        seg = seg.strip(" .,:;!\"'()[]-").strip()
        if len(seg)>= 2:
            parts.append(seg)

    cleaned, seen = [], set()
    for p in parts:
        c=re.sub(r"\s+", " ", p).strip()
        #cleans it by removing excess whitespace
        if 2<= len(c) <= 80:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(key)
                #we dont want duplicates

    return cleaned[:5]
    #returns only the top 5 candidates since we want to focus on mainly one winner
#def tweet_mentions_award(text:str, award: str) -> Tuple[bool, int]:
    #returns the index its found
    #i = text.lower().find(award.lower())
    #return (i >= 0, (i if i >= 0 else 10**9))

award_stop = {
    "best","by","an","a","the","or","of","in","for",
    "–","-","television","tv","series","motion","picture"
}

def tweet_mentions_award(text, award):
    #best and at least 2 meaningful words
    text_l = text.lower()
    if "best" not in text_l:
        return (False, 10**9)
    
    award_c = award.lower().replace("-", " ")
    words = [w for w in award_c.split() if w and w not in award_stop]
    matched = [w for w in words if w in text_l]
    if len(matched)<2:
        return(False, 10**9)
    
    index = min((text_l.find(w) for w in matched), default = 10**9)
    return (True, index)

award_lex = re.compile(r"\b(best|golden globe[s]?|award|goes to|wins?|won|category)\b", re.IGNORECASE)
cand_chars = r"[a-z0-9&'().,:!\-\s]{2,}"

pattern_goesto = re.compile(rf"\b(?:award\s+)?goes to\s+(?P<cand>{cand_chars})", re.IGNORECASE)
pattern_xwins  = re.compile(rf"(?P<cand>{cand_chars})\s+wins?\b", re.IGNORECASE)
pattern_wonfor = re.compile(rf"\bwon\b.*?\bfor\s+(?P<cand>{cand_chars})", re.IGNORECASE)

def clean_candidate(s:str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" .,:;!\"'()[]-")
    s = award_lex.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if "http" in s or s.startswith("@"):
        return ""
    return s

def get_x(text: str, cue_index: int) -> List[str]:
    window = set_window(text, cue_index)
    cands: List[str] = []

    for pat in (pattern_goesto, pattern_xwins, pattern_wonfor):
        for m in pat.finditer(window):
            c = clean_candidate(m.group("cand").lower())
            if 2<= len(c) <= 80:
                cands.append(c)

    if not cands:
        raw = " ".join(candidate_span.findall(window))
        for c in split_candidates(raw):
            cc = clean_candidate(c)
            if 2 <= len(cc) <= 80:
                cands.append(cc)

    final, seen = [], set()
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            final.append(c)

    return final[:5]

#main function
def find_winners (cleaned_path: str, awards: List[str], drop_retweets: bool=True) ->Dict[str, str]:
    #takes in tweets/awards - returns award name with possible winner
    scores: Dict[str, Counter] = {a: Counter() for a in awards}
    #for each award, keep a counter of candidate and their score

    count = 0
    for tweet in load_clean_tweets(cleaned_path):
        count += 1
    print(f"Loaded {count} tweets from {cleaned_path}")

    for tweet in load_clean_tweets(cleaned_path):
        if drop_retweets and tweet.get("is_retweet"):
            continue
        
        text = tweet.get("text", "")
        if not text:
            continue

        m = win_re.search(text)
        if not m:
            continue #doesnt have something to do with winning
        if negation_re.search(text) or future_re.search(text):
            continue #has something about not winnning or shouldve won

        candidates = get_x(text, m.start())
        if not candidates:
            continue
    
        for award in awards:
            found, a_index = tweet_mentions_award(text, award)
            if not found:
                continue
            
            dist = abs(a_index - m.start())
            base = pattern_weight(m.group(0))+ (2 if dist < 80 else 0) + (1 if dist < 140 else 0)

            for candidate in candidates:
                if len(candidate) <3:
                    continue
                scores[award][candidate] += base

    winners: Dict[str, str] = {}
    for award, counter in scores.items():
        if not counter:
            winners[award] = ""
            continue
        
        best = sorted(counter.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
        winners[award]= best
    
    return winners

