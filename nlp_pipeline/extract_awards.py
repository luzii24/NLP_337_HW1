'''
Answer (26)
best screenplay - motion picture
best director - motion picture
best performance by an actress in a television series - comedy or musical
best foreign language film
best performance by an actor in a supporting role in a motion picture
best performance by an actress in a supporting role in a series, mini-series or motion picture made for television
best motion picture - comedy or musical
best performance by an actress in a motion picture - comedy or musical
best mini-series or motion picture made for television
best original score - motion picture
best performance by an actress in a television series - drama
best performance by an actress in a motion picture - drama
cecil b. demille award
best performance by an actor in a motion picture - comedy or musical
best motion picture - drama
best performance by an actor in a supporting role in a series, mini-series or motion picture made for television
best performance by an actress in a supporting role in a motion picture
best television series - drama
best performance by an actor in a mini-series or motion picture made for television
best performance by an actress in a mini-series or motion picture made for television
best animated feature film
best original song - motion picture
best performance by an actor in a motion picture - drama
best television series - comedy or musical
best performance by an actor in a television series - drama
best performance by an actor in a television series - comedy or musical
'''
import re
import json
from collections import Counter
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet', quiet=True)
import spacy
from rapidfuzz import fuzz, process

award_patterns = [
    r"(?:award for|wins|won|receive[s]?|receiving|presenting|presented with|accept[s]?|accepting)\s+(?:the\s+)?(best [^.,;:!?]+)",
    r"(?:nominated for|nominee for|up for|contender for)\s+(?:the\s+)?(best [^.,;:!?]+)",
    r"(?:the\s+)?(best [^.,;:!?]+?)\s+(?:award\s+)?goes to",
    r"(?:the\s+)?(best [^.,;:!?]+?)\s+(?:is|was)\s+(?:won by|awarded to|received by)",
    r"(?:the\s+)?(best [^.,;:!?]+?)\s+(?:award|trophy|prize)",
    r"(?:for|in)\s+(?:the\s+)?(best [^.,;:!?]+)",
    r"(?:won|wins|winning)\s+(?:the\s+)?(best [^.,;:!?]+)",
    r"(?:congrats|congratulations|props)\s+(?:on|for)\s+(?:the\s+)?(best [^.,;:!?]+)",
    r"(?:the\s+)?award for\s+(?:the\s+)?(best [^.,;:!?]+)",
    r"\b(best [^-]+(?:-[^-]+)?)(?:\s*-\s*[^-]+){1,}"
]

def looks_like_winner_fragment(fragment):
    """Return True if fragment likely refers to a winner, not an award subcategory."""
    frag = fragment.strip()
    if not frag:
        return False
    doc = nlp(frag)

    # Named entities that suggest a person or creative work
    if any(ent.label_ in {"PERSON", "WORK_OF_ART"} for ent in doc.ents):
        return True

    # Contains multiple proper nouns (e.g. 'Maggie Smith')
    propn_count = sum(1 for t in doc if t.pos_ == "PROPN")
    if propn_count >= 2:
        return True

    # Single capitalized token that’s a PROPN (e.g. 'Skyfall')
    if len(doc) == 1 and doc[0].pos_ == "PROPN":
        return True

    return False

def extract(tweets):
    award_tweets = tweets[tweets.str.contains(r"\bbest\b", case=False, na=False)]

    print(f"Found {len(award_tweets)} tweets mentioning 'best'")
    
    award_counter = Counter()

    for text in tqdm(award_tweets, desc="Extracting awards"):
        # text_lower = text.lower()
        for pattern in award_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    m = [x for x in m if x][0]
                phrase = clean_award_phrase(m)
                if len(phrase.split()) >= 3 and phrase.startswith("best"):
                    award_counter[phrase] += 1
    return award_counter

def clean_award_phrase(phrase):
    """Clean up extracted award phrases."""
    phrase = phrase.lower().strip()
    phrase = re.split(r"\b(for|to|by|from|goes to|goes|at|is)\b", phrase)[0].strip()
    
    if "-" in phrase:
        parts = re.split(r"\s*-\s*", phrase)
        kept = []

        for i, part in enumerate(parts):
            if i == 0:
                kept.append(part.strip())
                continue
            if looks_like_winner_fragment(part):
                break  # Stop once we hit winner/title
            kept.append(part.strip())

        cleaned = " - ".join(p for p in kept if p)
        phrase = cleaned
            
    phrase = re.sub(r"[^a-zA-Z0-9\s\-]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()

    # Normalize "tv" → "television"
    phrase = re.sub(r"\btv\b", "television", phrase, flags=re.I)
    phrase = re.sub(r"\b(mini series|mini - series|mini-series)\b", "miniseries", phrase, flags=re.I)
    
    
    return phrase



nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet', quiet=True)

def remove_entities(phrase):
    phrase = phrase.lower().strip()
       
    # Normalize hyphens for NER, but keep original for output later
    doc = nlp(phrase)

    tokens = []
    for token in doc:
        # stop if a PERSON/WORK_OF_ART entity starts here
        if any(ent.label_ in {"PERSON", "WORK_OF_ART"} and ent.start <= token.i < ent.end for ent in doc.ents):
            break
        # stop if we hit a verb or interjection (indicating a new sentence/clause)
        if token.pos_ in {"INTJ", "PRON"}:
            break
        if token.pos_ == "ADP" and token.text not in {"in", "of"}:
            break
        tokens.append(token.text)

    phrase = " ".join(tokens).strip()    
    phrase = re.sub(r"\bminiseries\b", "mini-series", phrase, flags=re.I)
    
    if "in" in phrase:
        parts = re.split(r"\b(in)\b", phrase)
        if len(parts) <= 1:
            return phrase

        before = parts[0].strip()
        after = " ".join(parts[1:]).strip()
        after_words = re.findall(r"[a-z]+", after)[1:]
        # print(after_words)
        unknown_ratio = 0
        if after_words[1:]:
            unknown = [w for w in after_words if not wn.synsets(w)]
            # if unknown:
            #     print("Unknown", unknown)
            unknown_ratio = len(unknown) / len(after_words)
        if unknown_ratio >= 0.5:
            phrase = before.strip()
        else:
            phrase = phrase.strip()
    
    return phrase

def is_person_related(phrase):
    """Return True if the award likely relates to a person."""
    phrase = phrase[1:]
    for word in phrase.split():
        synsets = wn.synsets(word)
        if any("person" in s.lexname() for s in synsets):
            return True
    return False

def merge_similar_awards(counter, threshold):
    merged = {}
    seen = set()
    
    categories = {phrase: "person" if is_person_related(phrase) else "nonperson"
                  for phrase in counter}

    items = sorted(counter.items(), key=lambda x: -x[1])

    for phrase, _ in items:
        # if "score" not in phrase:
        #     continue
        # print(phrase)
        if phrase in seen:
            # print("seen")
            continue
        cat = categories[phrase]

        # find close matches
        matches = process.extract(phrase, counter.keys(), scorer=fuzz.token_set_ratio)
        group = [m for m, score, _ in matches if score >= threshold and categories[m] == cat]
        # DEBUG: show all high-similarity matches
        # print(f"--- Matches for: {phrase} ---")
        # for m, score, _ in matches:
        #     if score >= 70:  # show moderate matches too
        #         print(f"  {m} → {score}")

        # pick the most complete version (longest phrase)
        best_variant = max(group, key=len)
        total_count = sum(counter[g] for g in group)
        merged[best_variant] = total_count
        # print("-:", best_variant, "\n")
        for g in group:
            seen.add(g)
    # for k, v in sorted(merged.items(), key=lambda x: -x[1])[:20]:
    #     print(f"{k}: {v}")
    
    return merged



def refine_awards(raw_counter):
    # 1. Clean and filter
    cleaned = Counter()
    filtered = Counter({k: v for k, v in raw_counter.items() if v > 10})
    # filtered = Counter(dict(Counter(filtered).most_common(100)))
    # print(len(filtered))
    
    
    for phrase, count in tqdm(filtered.items(), desc="Refine awards"):
        phrase = remove_entities(phrase)
        if len(phrase.split()) < 2:
            continue
        cleaned[phrase] = count

    filtered = Counter(dict(Counter(cleaned).most_common(100)))

    merged = merge_similar_awards(filtered, threshold=85)

    return merged


def extract_awards(data_path):
    data = pd.read_json(data_path, lines=True)
    tweets = data["text"]
    # print(len(tweets))

    results = extract(tweets)
    # for phrase, count in results.most_common(30):
    #     print(f"{phrase}: {count}")

    award_candidates = results

    refined = refine_awards(award_candidates)
    # print("----- Award Results ------")
    # for k, v in sorted(refined.items(), key=lambda x: -x[1])[:30]:
    #     print(f"{k}: {v}")
                
    top_awards = [k for k, _ in sorted(refined.items(), key=lambda x: -x[1])[:30]]
    formatted = [a.strip() for a in top_awards]
    return formatted
    
