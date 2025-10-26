from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import List, Dict, Union

import spacy

# load spacy model once
NLP = spacy.load("en_core_web_sm")

Row = Union[str, dict]

def to_text(row: Row) -> str:
    # handle raw strings or dicts from pre-ceremony step
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        return row.get("text_original") or row.get("text") or ""
    return ""

def is_retweet(row: Row, low: str) -> bool:
    # cheap rt detector; ok to be imperfect
    if isinstance(row, dict) and row.get("is_retweet"):
        return True
    return low.startswith("rt ") or low.startswith("rt @")

def normalize(s: str) -> str:
    # simple text norm
    return re.sub(r"\s+", " ", s.lower().strip())

def clean_possessive(name: str) -> str:
    # turn "ben affleck's" into "ben affleck"
    return re.sub(r"[’']s\b", "", name).strip()

# drop obvious non-candidates that frequently swamp counts
BAD_FRAGMENTS = {
    "golden globes", "golden globe", "goldenglobes", "goldenglobe",
    "golden", "globe", "globes", "award", "awards", "best",
    "drama", "comedy", "musical", "television", "tv", "motion picture",
    "series", "rt", "gg", "hfpa", "red carpet", "host", "hosts"
}

# spans/phrases we never want to count as winners (too generic)
STOP_SPAN = {
    "best", "drama", "comedy", "musical", "motion", "picture", "motion picture",
    "television", "series", "original", "score", "song", "animated", "feature",
    "animated feature film", "foreign", "language", "foreign language film",
    "limited", "mini", "miniseries", "movie", "film", "award", "awards",
    "congratulations", "congrats", "hollywood", "hbo"
}

# explicit junk winners that sneak in via title-cased spans
BAD_WINNER_PHRASES = {
    "best actor", "best actress", "best television", "original score",
    "original song", "animated feature film", "foreign language film",
    "motion picture", "television series", "mini series", "limited series"
}

def clean_candidate(name: str, award_name: str) -> str:
    # strip possessives and cut trailing descriptors like " - best performance"
    n = clean_possessive(name)
    # cut at dashes or pipes and take the left chunk
    n = re.split(r"\s*[–—\-|]\s*", n)[0].strip()
    # remove stray quotes
    n = n.strip('"\''"“”‘’")
    # drop category words from titles so we don't count "best television series"
    drop = {
        "best","award","awards","motion","picture","television","series","comedy",
        "musical","drama","original","score","song","animated","feature","foreign",
        "language","limited","mini","miniseries"
    }
    tokens = re.findall(r"[a-zA-Z0-9'’.-]+", n)
    kept = [t for t in tokens if t.lower() not in drop]
    n = " ".join(kept).strip() or name.strip()
    return n

def slice_best_phrase(text_lower: str, max_tokens: int = 8) -> str | None:
    # pull a short "best ..." phrase for fuzzy award matching
    if "best" not in text_lower:
        return None
    toks = re.findall(r"[a-z0-9\-]+", text_lower)
    try:
        i = toks.index("best")
    except ValueError:
        return None
    boundary = {
        "award","awards","drama","comedy","musical","film","movie","director",
        "series","limited","animated","foreign","score","song"
    }
    out = []
    for j, tok in enumerate(toks[i:i+max_tokens]):
        out.append(tok)
        if tok in boundary and j >= 1:
            break
    return " ".join(out).strip() if out else None

def token_overlap(a: str, b: str) -> float:
    # quick overlap to shortlist an award
    ta = set(re.findall(r"[a-z0-9\-]+", normalize(a)))
    tb = set(re.findall(r"[a-z0-9\-]+", normalize(b)))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(tb))

def fuzzy_ratio(a: str, b: str) -> float:
    # small difflib-based ratio (kept simple to avoid extra deps)
    from difflib import SequenceMatcher
    return 100.0 * SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def match_award_in_tweet(text_lower: str, award_names: list[str]) -> str | None:
    # pick 1 best-matching award to avoid contamination
    # token overlap
    best, best_sc = None, 0.0
    for aw in award_names:
        sc = token_overlap(text_lower, aw)
        if sc > best_sc:
            best, best_sc = aw, sc
    if best_sc >= 0.55:
        return best

    # fuzzy on the "best .. " slice
    sl = slice_best_phrase(text_lower) or ""
    if sl:
        f_best, f_sc = None, -1.0
        for aw in award_names:
            sc = fuzzy_ratio(sl, aw)
            if sc > f_sc:
                f_best, f_sc = aw, sc
        if f_best and f_sc >= 72:
            return f_best

    return best if best_sc >= 0.35 else None

def is_bad_candidate(name: str, award_name: str) -> bool:
    # reject generic phrases and tokens that are basically the award text itself
    n = normalize(name)
    if n in BAD_FRAGMENTS or n.replace(" ", "") in {"goldenglobes", "goldenglobe"}:
        return True
    if "golden globe" in n or "golden globes" in n:
        return True
    aw_tokens = set(re.findall(r"[a-z0-9]+", normalize(award_name)))
    n_tokens = set(re.findall(r"[a-z0-9]+", n))
    if n_tokens and n_tokens.issubset(aw_tokens):
        return True
    return False

def is_person_award(award_name: str) -> bool:
    a = award_name.lower()
    return any(k in a for k in ["actor", "actress", "director"]) or "cecil b. demille" in a

def is_title_award(award_name: str) -> bool:
    a = award_name.lower()
    if "screenplay" in a or "song" in a or "score" in a:
        return False
    return any(k in a for k in ["motion picture", "film", "television series", "limited series", "animated", "foreign"])

def compile_award_regex(award: str) -> re.Pattern:
    # build a light regex that looks for key award tokens in order
    stop = {
        "the","best","by","an","a","in","of","or","-","–","—","and","any"
    }
    toks = [t for t in re.findall(r"[a-z0-9]+", award.lower()) if t not in stop]
    if not toks:
        toks = ["best"]
    pattern = r".*?".join(map(re.escape, toks))
    return re.compile(pattern, re.IGNORECASE)

def weight_from_text(low: str) -> float:
    # upweight tweets that look like live winner calls
    if any(p in low for p in [" wins ", " winner ", " goes to ", " award goes to ", " takes home "]):
        return 2.0
    if any(p in low for p in [" won ", "winning", "accepts", "accepted"]):
        return 1.5
    if "congrats" in low or "congratulations" in low:
        return 1.25
    return 1.0

def title_spans(doc) -> List[str]:
    # grab quoted text + title-cased spans but drop generic category phrases
    out = []
    out += re.findall(r"[\"“”‘’']([^\"“”‘’']+)[\"“”‘’']", doc.text)
    cur = []
    for tok in doc:
        if tok.is_space:
            if cur:
                out.append(" ".join(cur))
                cur = []
            continue
        if tok.text[:1].isupper():
            cur.append(tok.text)
            if len(cur) >= 6:
                out.append(" ".join(cur))
                cur = []
        else:
            if cur:
                out.append(" ".join(cur))
                cur = []
    if cur:
        out.append(" ".join(cur))
    # normalize + strip tiny fragments and drop stop spans
    cleaned = []
    for s in out:
        s2 = s.strip()
        if len(s2) <= 1:
            continue
        if re.fullmatch(r"[A-Z]{1,3}", s2):
            continue
        n = normalize(s2)
        if n in STOP_SPAN:
            continue
        if n in BAD_WINNER_PHRASES:
            continue
        if "best " in n:
            continue
        cleaned.append(s2)
    return cleaned

# define main
def extract_winners(
    tweets: List[Row],
    award_names: List[str],
    debug: bool = False,
) -> Dict[str, str]:
    """
    return a mapping award_name -> single winner string.
    - tweets may be raw strings or dicts with 'text'/'text_original'
    - award_names are the official categories (lowercase is fine)
    """
    nlp = NLP

    # only strong winner triggers
    must_have = (" wins ", " won ", " goes to ", " award goes to ", " takes home ", " is awarded to ")

    # counters per award
    tallies: Dict[str, Counter] = {aw: Counter() for aw in award_names}

    for row in tweets:
        text = to_text(row)
        if not text:
            continue
        low = text.lower()

        if not any(k in low for k in must_have):
            continue  # skip general chatter

        matched_award = match_award_in_tweet(low, award_names)
        if not matched_award:
            continue  # skip if we can't confidently map this tweet to a single award

        # parse once
        doc = nlp(text)
        people = {ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"}
        titles = set(title_spans(doc))

        base_w = weight_from_text(low) * (0.5 if is_retweet(row, low) else 1.0)

        # per-tweet de-dupe so a name only counts once per tweet
        seen = set()

        if is_person_award(matched_award):
            for p in people:
                name = clean_candidate(p, matched_award)
                if not name or name in seen:
                    continue
                if is_bad_candidate(name, matched_award):
                    continue
                if normalize(name) in BAD_WINNER_PHRASES:
                    continue
                if len(name) >= 3 and re.search(r"[A-Za-z]", name):
                    tallies[matched_award][name] += base_w
                    seen.add(name)
        elif is_title_award(matched_award):
            for t in titles:
                title = clean_candidate(t, matched_award)
                if not title or title in seen:
                    continue
                if is_bad_candidate(title, matched_award):
                    continue
                if normalize(title) in BAD_WINNER_PHRASES:
                    continue
                if len(title) >= 2 and re.search(r"[A-Za-z]", title):
                    tallies[matched_award][title] += base_w
                    seen.add(title)
        else:
            for cand in list(people) + list(titles):
                c = clean_candidate(cand, matched_award)
                if not c or c in seen:
                    continue
                if is_bad_candidate(c, matched_award):
                    continue
                if normalize(c) in BAD_WINNER_PHRASES:
                    continue
                if len(c) >= 3 and re.search(r"[A-Za-z]", c):
                    tallies[matched_award][c] += base_w
                    seen.add(c)

    # pick top for each award; empty if none
    winners: Dict[str, str] = {}
    for aw, cnt in tallies.items():
        winners[aw] = cnt.most_common(1)[0][0] if cnt else ""

    return winners
