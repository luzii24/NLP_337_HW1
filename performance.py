import re
import pandas as pd
import spacy
from tqdm import tqdm

def get_performance():
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_json("tweets_cleaned.jsonl", lines=True)

    performance_keywords = [
        r"\bperformance\b", r"\bperform\b", r"\bsing\b", r"\bmonologue\b", r"\bspeech\b"
    ]

    performance_pattern = re.compile("|".join(performance_keywords), re.IGNORECASE)

    df["is_performance"] = df["text"].str.contains(performance_pattern)
    performance_tweets = df[df["is_performance"]]["text"].tolist()
    print(f"Performance-related tweets: {len(performance_tweets)}")


    def extract_entities(tweets, keyword_regex, context_label):
        results = []
        for doc in nlp.pipe(tweets, batch_size=50):
            text = doc.text
            if not keyword_regex.search(text):
                continue
            ents = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
            if ents:
                results.append({
                    "context": context_label,
                    "entities": ", ".join(sorted(set(ents))),
                    "text": text
                })
        return results

    performance_records = extract_entities(performance_tweets, performance_pattern, "performance")
    performance_df = pd.DataFrame(performance_records)


    def summarize_mentions(df, label):
        if df.empty:
            return pd.DataFrame()
        summary = (
            df.assign(entity=df["entities"].str.split(", "))
            .explode("entity")
            .groupby("entity")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        summary["context"] = label
        return summary

    performance_summary = summarize_mentions(performance_df, "performance")


    def clean_entity(name):
        name = name.strip()
        name = name.title()
        if name.lower().startswith("omg "):
            name = name[4:]
        name = name.replace(" - ", " ").replace("-", " ").strip()
        return name

    performance_summary["clean_entity"] = performance_summary["entity"].apply(clean_entity)

    clean_entities = performance_summary["clean_entity"].drop_duplicates().tolist()

    # remove single-name entities that are contained in longer ones
    filtered_entities = []
    for ent in clean_entities:
        if not any(ent != other and ent in other for other in clean_entities):
            filtered_entities.append(ent)

    cleaned_summary = (
        performance_summary.groupby("clean_entity", as_index=False)
            .agg({"count": "sum", "context": "first"})
            .sort_values("count", ascending=False)
    )

    top_entities = cleaned_summary["clean_entity"].head(8).tolist()
    print("\nMost mentioned performers and speakers:")
    print(top_entities)