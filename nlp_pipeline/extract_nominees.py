from __future__ import annotations

import json
import re
from collections import Counter
from typing import List, Dict

import spacy

from difflib import SequenceMatcher

# quick filters to keep garbage out of candidates (keep lowercase)
STOP_SPAN = {
    "best", "golden", "globe", "globes", "award", "awards",
    "drama", "comedy", "musical", "series", "television",
    "actor", "actress", "wins", "winner", "nominee", "nominated"
}

BAD_CAND_SUBSTR = (
    "@", "http://", "https://", "www.", "pic.twitter", "#"
)

# load spacy model once
_NLP = spacy.load("en_core_web_sm")

def to_text(row) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        return row.get("text") or row.get("text_original") or ""
    return ""

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def token_set(s: str) -> set:
    return set(re.findall(r"[a-z0-9\-]+", normalize(s)))

def slice_best_phrase(text_lower: str, max_tokens: int = 8) -> str | None:
    # best slice
    if "best" not in text_lower:
        return None
    toks = re.findall(r"[a-z0-9\-]+", text_lower)
    try:
        start = toks.index("best")
    except ValueError:
        return None

    stop_words = {
        "award", "awards", "drama", "comedy", "musical", "film", "movie",
        "director", "series", "limited", "animated", "foreign", "score", "song"
    }

    out = []
    for i, tok in enumerate(toks[start:start + max_tokens]):
        out.append(tok)
        if tok in stop_words and i >= 1:
            break
    return " ".join(out).strip() if out else None


def is_person_award(name: str) -> bool:
    a = name.lower()
    return any(k in a for k in ["actor", "actress", "director", "screenplay", "cecil b. demille"])  # person-centric


def token_overlap(a: str, b: str) -> float:
    # simple overlap to pick the closest award
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(tb))


def titlecase_spans(doc) -> list[str]:
    # collect 1-5 token title-cased spans, useful for movie/show names
    out, cur = [], []
    for tok in doc:
        if tok.is_space:
            if cur:
                txt = " ".join(cur)
                # skip if the span contains category words like "best", "award" etc
                if not any(w.lower() in STOP_SPAN for w in txt.split()):
                    out.append(txt)
                cur = []
            continue
        t = tok.text
        if t[:1].isupper() and not t.isupper():  # avoid all-caps shoutout
            cur.append(t)
            if len(cur) >= 5:
                txt = " ".join(cur)
                if not any(w.lower() in STOP_SPAN for w in txt.split()):
                    out.append(txt)
                cur = []
        else:
            if cur:
                txt = " ".join(cur)
                if not any(w.lower() in STOP_SPAN for w in txt.split()):
                    out.append(txt)
                cur = []
    if cur:
        txt = " ".join(cur)
        if not any(w.lower() in STOP_SPAN for w in txt.split()):
            out.append(txt)
    return out


def extract_candidates(doc, award_name: str) -> set[str]:
    # ner first, then fallback to quotes and titlecase spans
    people, titles = set(), set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.add(ent.text.strip())
        elif ent.label_ in {"WORK_OF_ART", "ORG"}:
            titles.add(ent.text.strip())

    for m in re.findall(r"[\"“”‘’']([^\"“”‘’']+)[\"“”‘’']", doc.text):
        titles.add(m.strip())

    for span in titlecase_spans(doc):
        titles.add(span.strip())

    if is_person_award(award_name):
        return {p for p in people if p}
    else:
        return {t for t in titles if t}

def clean_candidate(name: str, award_name: str) -> str | None:
    s = name.strip()
    if not s:
        return None
    # drop if looks like a handle etc
    low = s.lower()
    if any(b in low for b in BAD_CAND_SUBSTR):
        return None
    # strip possessives and obvious tail noise 
    s = re.sub(r"[’']s\b", "", s)
    s = re.split(r"\s*[-–—]\s*", s)[0].strip()
    # trim stray punctuation/dashes
    s = s.strip(" -–—|,.;:!?\"'“”‘’")
    if not s:
        return None

    if is_person_award(award_name):
        # keep likely first + last 
        parts = s.split()
        if len(parts) < 2:
            return None
        keep = parts[:2]
        # include a third token if hyphenated surname piece (Day-Lewis)
        if len(parts) >= 3 and ("-" in parts[2] or parts[2][0:1].isupper()):
            keep.append(parts[2])
        cleaned = " ".join(keep).strip()
        # start with capital
        if not all(p[:1].isupper() for p in keep[:2]):
            return None
        return cleaned
    else:
        # drop category words 
        words = [w for w in s.split() if w.lower() not in STOP_SPAN]
        cleaned = " ".join(words).strip()
        if not cleaned:
            return None
        # avoid returning generic words
        if cleaned.lower() in {"best", "drama", "comedy", "musical", "series"}:
            return None
        return cleaned

# define main
def extract_nominees(
    tweets: List[str | dict],
    award_names: List[str],
    top_k: int = 4,
    debug: bool = False,
) -> Dict[str, List[str]]:
    """fast, no-internet nominees extractor. keeps it simple and quick.

    tweets: list of tweet strings or dicts with text/text_original
    award_names: list of canonical award names (lowercase, hyphens ok)
    top_k: return this many per award (default 4)
    """
    # verb hints
    hints = (
        "best ", " nominee", " nomin", "should win", "should have won", "wins", "won"
    )
    drop_if = ("dress", "red carpet", "monologue")

    # per award counters
    buckets: Dict[str, Counter] = {aw: Counter() for aw in award_names}

    def ratio(a: str, b: str) -> float:
        # difflib ratio on normalized strings
        return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

    for row in tweets:
        text = to_text(row)
        if not text:
            continue
        low = text.lower()

        # fast relevance gate
        if not any(h in low for h in hints):
            continue
        if any(bad in low for bad in drop_if):
            continue

        phrase = slice_best_phrase(low, max_tokens=8) or low
        # combine token overlap with a light difflib ratio
        scores = []
        for a in award_names:
            sc = 0.7 * token_overlap(phrase, a) + 0.3 * ratio(phrase, a)
            scores.append((sc, a))
        best_sc, best_aw = max(scores, key=lambda x: x[0])
        if best_sc < 0.35:
            continue

        # clean and count
        doc = _NLP(text)
        seen_this_tweet = set()
        for cand in extract_candidates(doc, best_aw):
            cleaned = clean_candidate(cand, best_aw)
            if not cleaned:
                continue
            key = normalize(cleaned)
            if key in seen_this_tweet:
                continue
            seen_this_tweet.add(key)
            buckets[best_aw][cleaned] += 1
            if debug:
                print(f"[{best_aw}] +1 :: {cleaned}")

    results: Dict[str, List[str]] = {}
    for aw in award_names:
        merged, surface = Counter(), {}
        for name, cnt in buckets[aw].items():
            key = normalize(name)
            merged[key] += cnt
            surface.setdefault(key, name)
        results[aw] = [surface[k] for k, _ in merged.most_common(top_k)]
    return results
