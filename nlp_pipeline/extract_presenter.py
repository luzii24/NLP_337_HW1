import spacy
import re
from collections import defaultdict
import ftfy
from unidecode import unidecode
from langdetect import detect
import inflection
import nltk
from nltk import word_tokenize, pos_tag
from difflib import get_close_matches
from imdb import Cinemagoer
from functools import lru_cache
from tqdm import tqdm
import json
from rapidfuzz import fuzz, process
ia = Cinemagoer()

HARD_AWARD_CATEGORIES = {
    "best screenplay - motion picture",
    "best director - motion picture",
    "best performance by an actress in a television series - comedy or musical",
    "best foreign language film",
    "best performance by an actor in a supporting role in a motion picture",
    "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
    "best motion picture - comedy or musical",
    "best performance by an actress in a motion picture - comedy or musical",
    "best mini-series or motion picture made for television",
    "best original score - motion picture",
    "best performance by an actress in a television series - drama",
    "best performance by an actress in a motion picture - drama",
    "cecil b. demille award",
    "best performance by an actor in a motion picture - comedy or musical",
    "best motion picture - drama",
    "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
    "best performance by an actress in a supporting role in a motion picture",
    "best television series - drama",
    "best performance by an actor in a mini-series or motion picture made for television",
    "best performance by an actress in a mini-series or motion picture made for television",
    "best animated feature film",
    "best original song - motion picture",
    "best performance by an actor in a motion picture - drama",
    "best television series - comedy or musical",
    "best performance by an actor in a television series - drama",
    "best performance by an actor in a television series - comedy or musical"
}

# Helper functions to validate real movies and people using Cinemagoer

@lru_cache(maxsize=10000)
def is_real_person(name):
    try:
        results = ia.search_person(name)
        return bool(results)
    except:
        return False

def normalize_terms(text):
    synonyms = {
        "tv": "television",
        "television": "television"
    }
    words = set(text.lower().split())
    normalized = set(synonyms.get(w, w) for w in words)
    return normalized

def best_award_match(extracted_phrase, official_awards):
    if not extracted_phrase:
        return None

    STOP_WORDS = {"award", "awards", "goes", "won", "to"}
    important_terms = {"actor", "actress", "supporting", "song", "score", "television", "TV", "screenplay", "series"}
    words = extracted_phrase.split()
    for i, w in enumerate(words):
        if w.lower() in STOP_WORDS:
            extracted_phrase = " ".join(words[:i])
            break
    remove = r"\b(?:{})\b.*".format("|".join(map(re.escape, STOP_WORDS)))
    extracted_phrase = re.sub(remove, "", extracted_phrase).strip()

    scores = []
    for award in official_awards:
        base_score = fuzz.token_sort_ratio(extracted_phrase, award.lower())
        # --- semantic weighting (boost if key words match) ---
        
        extracted_terms = normalize_terms(extracted_phrase)
        award_terms = normalize_terms(award)
        overlap = important_terms.intersection(extracted_terms & award_terms)
        if overlap:
            base_score += 10 * len(overlap)  # boost for each important word match

        scores.append((award, base_score))

    best_match, best_score = max(scores, key=lambda x: x[1])

    if best_score >= 70:  # adjust threshold
        return best_match
    else:
        if best_score >= 65:
            print(f"No good match for: {extracted_phrase} (best {best_score}) match to {best_match}")
        return None

def find_best_imdb_match(name, top_n=3):
    """Find the most similar IMDb person name for a given name."""
    try:
        results = ia.search_person(name)
    except Exception as e:
        print(f"IMDb error for {name}: {e}")
        return None
    
    if not results:
        return None

    candidate_names = [p['name'] for p in results]
    best_match = process.extractOne(name, candidate_names, scorer=fuzz.WRatio)
    if best_match and best_match[1] > 85:  # confidence threshold
        return best_match[0]
    return None

def merge_partial_names(merged):
    names = list(merged.keys())
    to_merge = {}
    
    for name1 in names:
        for name2 in names:
            if name1 == name2:
                continue
            # If one name is contained within another (case-insensitive)
            if name1.lower() in name2.lower() or name2.lower() in name1.lower():
                # Merge to the longer name (assumed more complete)
                longer = name1 if len(name1) > len(name2) else name2
                shorter = name2 if longer == name1 else name1
                to_merge[shorter] = longer

    # Apply merges
    for short, long in to_merge.items():
        if short in merged and long in merged:
            merged[long].extend(merged[short])
            del merged[short]

    return merged

def merge_similar_names_by_award(results_dict):
    """Merge similar names per award, only keeping IMDb-verified ones."""
    merged_results = {}

    for award, names in results_dict.items():
        merged = defaultdict(list)
        removed = []

        for name in names:
            best_imdb_name = find_best_imdb_match(name)
            
            if not best_imdb_name:
                removed.append(name)
                continue  # skip this name entirely
            
            merged[best_imdb_name].append(name)
        merged = merge_partial_names(merged)
        
        # Only keep IMDb-confirmed names
        cleaned_names = list(merged.keys())
        merged_results[award] = cleaned_names

        for imdb_name, raw_variants in merged.items():
            if len(raw_variants) > 1:
                print(f"Merged {raw_variants} -> {imdb_name}")
        if removed:
            print(f"Removed (not found on IMDb): {removed}")

    return merged_results

nlp = spacy.load("en_core_web_sm")

def extract_presenters(tweets, award_names):
    '''
    Given cleaned tweets and known award names, return a dictionary mapping
    each award to a list of nominee names extracted from tweets.
    '''
    presenters = defaultdict(set)

    # clean and normalize tweets
    
    cleaned_tweets = []
    for tweet in tqdm(tweets, desc="Processing data"):
        try:
            fixed = ftfy.fix_text(tweet)
            normalized = unidecode(fixed)
            cleaned_tweets.append(normalized)
        except:
            continue
   
    keywords = ["present", "announce", "read", "introduce", "give",]
    filtered_tweets = [t for t in cleaned_tweets if any(kw in t.lower() for kw in keywords)]
    
    print(f"Filtered down to {len(filtered_tweets)} presenter-related tweets")
    output_file = 'presenter_related.json'
    with open(output_file, "w") as f:
        json.dump(filtered_tweets, f, indent=2)
    print(f"Saved {len(filtered_tweets)} tweets to {output_file}")
    
    # with open("presenter_related.json", "r") as f:
    #     filtered_tweets = json.load(f)

    presenter_patterns = [
        r"([\w\s,&]+?)\s+(?:present|announce|introduce|give|hand)\w*\s+(?:the\s+)?(best\s+[^.,;:!?]+)",
        r"([\w\s,&]+?)\s+(?:is|are)\s+(?:presenting|introducing|announcing)\s+(?:the\s+)?(best\s+[^.,;:!?]+)",
        r"([\w\s,&]+?)\s+(?:present|announce|introduce|give|hand)\w*\s+(?:the\s+award\s+)?for\s+(best\s+[^.,;:!?]+)",
        r"(?:presented by|announced by|introduced by|hosted by)\s+([\w\s,&]+)",
        r"([\w\s,&]+?)\s+(?:present|announce|introduce|give|hand)\w*\s+(?:the\s+(?:nominees\s+for\s+)?|for\s+)?(best\s+[^.,;:!?]+)",
        r"([\w\s,&]+?)\s+(?:present|announce|introduce|give|hand)\w*\s+(?:the\s+)?(cecil b(?:\.|)? demille(?: award)?)"
    ]
    
    presenter_STOPWORDS = {"As", "They", "Are", "While", "When"}
    

    # for tweet in tqdm(filtered_tweets, desc="Parsing"):
    for tweet in filtered_tweets:
        tweet_lower = tweet.lower()

        for pattern in presenter_patterns:
            match = re.search(pattern, tweet_lower)
            if match:
                if len(match.groups()) == 2:
                    p_raw, award_raw = match.groups()
                else:
                    p_raw, award_raw = match.group(1), None
                
                potential_names = re.split(r"\band\b|,|&", p_raw)
                clean_names = []
                for name in potential_names:
                    name = name.strip().title()
                    if len(name.split()) > 4:
                        remove = r"\b(?:{})\b.*".format("|".join(map(re.escape, presenter_STOPWORDS)))
                        name = re.sub(remove, "", name).strip()
                    if len(name.split()) > 0 and len(name.split()) < 5:
                        doc = nlp(name)
                        if len(doc.ents) == 0:
                            if all(tok.pos_ in {"PROPN", "NOUN"} for tok in doc):
                                clean_names.append(name)                      
                        for ent in doc.ents:
                            if ent.label_ == "PERSON":
                                clean_names.append(ent.text)
                            else:
                                print(f"{ent.text} Not name")
                                
                if clean_names:
                    if award_raw:
                        # award_raw = clean_award_phrase(award_raw)
                        matched_award = best_award_match(award_raw, HARD_AWARD_CATEGORIES)
                    else:
                        matched_award = None

                    if matched_award:
                        presenters[matched_award].update(clean_names)

                    

        
    results = merge_similar_names_by_award(presenters)
    for award, names in results.items():
        print(f"{award}: {names}")

    # return {award: sorted(set(presenters.get(award, []))) for award in award_names}
    return {award: sorted(list(names)) for award, names in results.items()}


