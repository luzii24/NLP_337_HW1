import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# make sure the lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# labels for overall verdict 
LABELS = [
    "dumpster fire",
    "terrible",
    "bad",
    "not great",
    "fine",
    "pretty decent",
    "good",
    "really good",
    "excellent",
    "fantastic",
    "iconic",
]

def analyze_sentiment(tweets_path="tweets_cleaned.jsonl", out_path="sentiment_summary.json"):
    rows = []
    with open(tweets_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    sid = SentimentIntensityAnalyzer()

    pos = neg = neu = 0
    very_pos = very_neg = 0
    pos_sum = 0.0
    neg_sum = 0.0
    comp_sum = 0.0

    for r in rows:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        c = sid.polarity_scores(txt)["compound"]
        comp_sum += c
        if c > 0.05:
            pos += 1
            pos_sum += c
            if c >= 0.5:
                very_pos += 1
        elif c < -0.05:
            neg += 1
            neg_sum += c
            if c <= -0.5:
                very_neg += 1
        else:
            neu += 1

    total = pos + neg + neu
    avg_pos = round(pos_sum / pos, 4) if pos else 0.0
    avg_neg = round(neg_sum / neg, 4) if neg else 0.0
    avg_comp = round((comp_sum / total) if total else 0.0, 4)

    # verdict by positive share 
    if total:
        pos_share = pos / total
        bucket = max(0, min(10, int(round(pos_share * 10))))
    else:
        bucket = 5
    verdict = LABELS[bucket]

    out = {
        "total_scored": total,
        "positive_count": pos,
        "neutral_count": neu,
        "negative_count": neg,
        "very_positive_count": very_pos,
        "very_negative_count": very_neg,
        "avg_positive": avg_pos,
        "avg_negative": avg_neg,
        "avg_compound": avg_comp,
        "pos_share": round((pos / total) if total else 0.0, 4),
        "neg_share": round((neg / total) if total else 0.0, 4),
        "neu_share": round((neu / total) if total else 0.0, 4),
        "verdict": verdict,
        "source_file": tweets_path,
        "sample": "all",
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    analyze_sentiment()