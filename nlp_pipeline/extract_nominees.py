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
def is_real_movie(title):
    try:
        results = ia.search_movie(title)
        return bool(results and results[0].get('kind') in {'movie', 'tv series', 'tv mini series'})
    except:
        return False

@lru_cache(maxsize=10000)
def is_real_person(name):
    try:
        results = ia.search_person(name)
        return bool(results)
    except:
        return False

nlp = spacy.load("en_core_web_sm")


def extract_nominees(tweets, award_names):
    '''
    Given cleaned tweets and known award names, return a dictionary mapping
    each award to a list of nominee names extracted from tweets.
    '''
    nominees = defaultdict(set)

    # clean and normalize tweets
    cleaned_tweets = []
    for tweet in tweets:
        try:
            fixed = ftfy.fix_text(tweet)
            normalized = unidecode(fixed)
            if detect(normalized) == 'en':
                cleaned_tweets.append(normalized)
        except:
            continue

    # filter tweets with nominee-related keywords
    keywords = ["nominated", "nominees", "deserves", "should win", "goes to", "hope", "nominee"]
    filtered_tweets = [tweet for tweet in cleaned_tweets if any(kw in tweet.lower() for kw in keywords)]
    print(f"\U0001f9f9 Filtered down to {len(filtered_tweets)} relevant tweets")

    # define nominee-related regex patterns
    nominee_patterns = [
        r"(?:nominated for|nominees for|is nominated for|are nominated for)\s+(.*?)(?:[.,;:!?]|$)",
        r"(?:hope|should|deserves|goes to)\s+(.*?)(?:[.,;:!?]|$)",
        r"(.*?)\s+is\s+nominated",
        r"(.*?)\s+(?:wins|won|takes|gets|received)\s+(.*?)"
    ]

    print(f"\U0001f9ea Running nominee extraction on {len(tweets)} tweets")

    for tweet in filtered_tweets:
        tweet_lower = tweet.lower()
        doc = nlp(tweet)

        for pattern in nominee_patterns:
            matches = re.findall(pattern, tweet_lower)
            if not matches:
                continue

            if isinstance(matches[0], tuple):
                matched_chunks = [item for pair in matches for item in pair]
            else:
                matched_chunks = matches

            for chunk in matched_chunks:
                chunk_doc = nlp(chunk)
                print(f"\u2705 Match: {chunk}")
                print(f"\U0001f50e Entities: {[ent.text for ent in chunk_doc.ents]}")

                # named entity recognition
                extracted_entities = [ent.text.strip() for ent in chunk_doc.ents if ent.label_ in {"PERSON", "WORK_OF_ART", "ORG"}]

                # POS tagging
                tokens = word_tokenize(chunk)
                pos_tags = pos_tag(tokens)
                pos_names = [word for word, tag in pos_tags if tag in {"NNP", "NN"}]

                # garbage filters
                garbage_phrases = {"doesn", "lol", "xo", "the year", "next year", "this year", "one", "two", "today", "tomorrow", "tonight"}
                cleaned_entities = set()

                for ent in extracted_entities:
                    ent_clean = ent.lower().strip()
                    if ent_clean not in garbage_phrases and len(ent_clean) > 2 and not ent_clean.isdigit():
                        ent_title = ent.title().replace("’s", "").replace("'s", "").strip()
                        if ent_title.lower() in {"congrats", "win", "won", "award", "globe"}:
                            continue
                        if "http" in ent_title.lower() or "@" in ent_title:
                            continue
                        cleaned_entities.add(ent_title)

                for name in pos_names:
                    name_clean = name.strip().title()
                    if len(name_clean) > 2 and name_clean.lower() not in garbage_phrases:
                        cleaned_entities.add(name_clean)

                # merge close matches
                deduped_entities = set()
                for ent in cleaned_entities:
                    match = get_close_matches(ent, deduped_entities, cutoff=0.85)
                    if match:
                        deduped_entities.add(match[0])
                    else:
                        deduped_entities.add(ent)

                # associate entities with awards
                for ent_text in deduped_entities:
                    # IMDb validation filter
                    for award in HARD_AWARD_CATEGORIES:
                        award_words = set(inflection.singularize(word) for word in award.lower().split()) - {"best", "by", "in", "a", "the", "of", "-", "or"}
                        match_count = sum(1 for word in award_words if word in tweet_lower)
                        if match_count >= 2:
                            # entity-type-aware validation
                            if (
                                ("motion picture" in award or "film" in award or "movie" in award) and is_real_movie(ent_text)
                            ) or (
                                ("actress" in award or "actor" in award or "cecil" in award or "director" in award) and is_real_person(ent_text)
                            ):
                                nominees[award].add(ent_text)

    for award, names in nominees.items():
        print(f"\U0001f3c6 Final nominees for {award}: {names}")

    return {award: sorted(set(nominees.get(award, []))) for award in HARD_AWARD_CATEGORIES}