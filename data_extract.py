import json
import pandas as pd
import re
import datetime
from ftfy import fix_text
from unidecode import unidecode
from langdetect import detect, detect_langs
from tqdm import tqdm

def is_english(text, threshold=0.5):
    if not text or text.strip() == "":
        return False
    try:
        langs = detect_langs(text)
        if not langs:
            return False
        top = langs[0]
        # result = top.lang == 'en' and top.prob >= threshold
        # if not result:
        #     print(text)
        #     print(langs[0])
        return top.lang == 'en' and top.prob >= threshold
    except Exception:
        return False

# Clean text
def clean_text(text):
    text = fix_text(text)
    text = unidecode(text)
    text = re.sub(r"http\S+", "", text)                 # remove URLs
    text = re.sub(r"(@\w+)|(#\w+)|\brt\b", "", text)    # remove mentions, hashtags, RT     
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)         # remove punctuation
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(' +', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_retweet(text):
    if not text:
        return {"is_retweet": False, "is_quote": False, "original_author": None, "original_text": "", "user_comment": ""}

    # simple retweet
    m = re.match(r"^RT @(\w+):\s*(.*)", text)
    if m:
        author = m.group(1)
        original = m.group(2)
        return {
            "is_retweet": True,
            "is_quote": False,
            "original_author": author,
            "original_text": clean_text(original),
            "user_comment": ""
        }
    
    # Quote tweet (contains " RT @user:")
    m = re.search(r"\sRT @(\w+):\s*(.*)", text)
    if m:
        author = m.group(1)
        original = m.group(2)
        user_comment = text[:m.start()].strip()
        return {
            "is_retweet": False,
            "is_quote": True,
            "original_author": author,
            "original_text": clean_text(original),
            "user_comment": user_comment
        }
        
    # Normal tweet    
    return {"is_retweet": False, "is_quote": False, "original_author": None, "original_text": "", "user_comment": ""}




if __name__ == "__main__":

    file_path = "gg2013.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    ## Store in jsonl
    output_file = "tweets_cleaned.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tweet_num = len(data)
    tweets = []
    for t in tqdm(data, desc="Pre-processing: "):
        
        text = t["text"]
        retweet_info = parse_retweet(text)
        
        if retweet_info["is_retweet"]:
            cleaned = retweet_info["original_text"]
        elif retweet_info["is_quote"]:
            text_to_clean = retweet_info["user_comment"]
            cleaned = clean_text(text_to_clean)
        else:
            cleaned = clean_text(text)
        
        hashtags = re.findall(r"#(\w+)", text)  # extract hashtags without '#'
        
        # Skip if not English or empty
        if not is_english(cleaned):
            continue
        
        tweets.append({
            "timestamp": datetime.datetime.fromtimestamp(t["timestamp_ms"]/1000.0).isoformat(),
            "screen_name": t["user"]["screen_name"],
            "user_id": t["user"]["id"],
            "text": cleaned,
            "hashtags": hashtags,
            **retweet_info
        })

    df = pd.DataFrame(tweets)
    # df = df.sort_values(by="timestamp", ascending=True)
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(df)} tweets to {output_file}")
            
