'''Version 0.5'''

# Year of the Golden Globes ceremony being analyzed
YEAR = "2013"

# Global variable for hardcoded award names
# This list is used by get_nominees(), get_winner(), and get_presenters() functions
# as the keys for their returned dictionaries
# Students should populate this list with the actual award categories for their year, to avoid cascading errors on outputs that depend on correctly extracting award names (e.g., nominees, presenters, winner)
AWARD_NAMES = [
    "best motion picture - drama",
    "best motion picture - comedy or musical",
    "best performance by an actor in a motion picture - drama",
    # Add or modify categories as needed for your year
    "your custom award category",
    # ... etc
]

def get_hosts(year):

    '''Returns the host(s) of the Golden Globes ceremony for the given year.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        list: A list of strings containing the host names. 
              Example: ["Seth Meyers"] or ["Tina Fey", "Amy Poehler"]
    
    Note:
        - Do NOT change the name of this function or what it returns
        - The function should return a list even if there's only one host
    '''
    # Your code here
    from hosts import find_hosts
    cleaned_path = "tweets_cleaned.jsonl"
    hosts = find_hosts(cleaned_path)

    return hosts

def get_awards(year):
    '''Returns the list of award categories for the Golden Globes ceremony.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        list: A list of strings containing award category names.
              Example: ["Best Motion Picture - Drama", "Best Motion Picture - Musical or Comedy", 
                       "Best Performance by an Actor in a Motion Picture - Drama"]
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Award names should be extracted from tweets, not hardcoded
        - The only hardcoded part allowed is the word "Best"
    '''
    # Your code here
    return awards

def get_nominees(year):
    '''Returns the nominees for each award category.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        dict: A dictionary where keys are award category names and values are 
              lists of nominee strings.
              Example: {
                  "Best Motion Picture - Drama": [
                      "Three Billboards Outside Ebbing, Missouri",
                      "Call Me by Your Name", 
                      "Dunkirk",
                      "The Post",
                      "The Shape of Water"
                  ],
                  "Best Motion Picture - Musical or Comedy": [
                      "Lady Bird",
                      "The Disaster Artist",
                      "Get Out",
                      "The Greatest Showman",
                      "I, Tonya"
                  ]
              }
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Use the hardcoded award names as keys (from the global AWARD_NAMES list)
        - Each value should be a list of strings, even if there's only one nominee
    '''
    # Your code here
    return nominees

def get_winner(year):
    '''Returns the winner for each award category.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        dict: A dictionary where keys are award category names and values are 
              single winner strings.
              Example: {
                  "Best Motion Picture - Drama": "Three Billboards Outside Ebbing, Missouri",
                  "Best Motion Picture - Musical or Comedy": "Lady Bird",
                  "Best Performance by an Actor in a Motion Picture - Drama": "Gary Oldman"
              }
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Use the hardcoded award names as keys (from the global AWARD_NAMES list)
        - Each value should be a single string (the winner's name)
    '''
    # Your code here
    return winners

def get_presenters(year):
    '''Returns the presenters for each award category.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        dict: A dictionary where keys are award category names and values are 
              lists of presenter strings.
              Example: {
                  "Best Motion Picture - Drama": ["Barbra Streisand"],
                  "Best Motion Picture - Musical or Comedy": ["Alicia Vikander", "Michael Keaton"],
                  "Best Performance by an Actor in a Motion Picture - Drama": ["Emma Stone"]
              }
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Use the hardcoded award names as keys (from the global AWARD_NAMES list)
        - Each value should be a list of strings, even if there's only one presenter
    '''
    # Your code here
    return presenters

def pre_ceremony():
    '''Pre-processes and loads data for the Golden Globes analysis.
    
    This function should be called before any other functions to:
    - Load and process the tweet data from gg2013.json
    - Download required models (e.g., spaCy models)
    - Perform any initial data cleaning or preprocessing
    - Store processed data in files or database for later use
    
    This is the first function the TA will run when grading.
    
    Note:
        - Do NOT change the name of this function or what it returns
        - This function should handle all one-time setup tasks
        - Print progress messages to help with debugging
    '''
    import io
    import json
    import re
    import zipfile
    from datetime import datetime
    from ftfy import fix_text
    import unidecode as _unidecode

    # look for raw data in the current directory
    input_candidates = ["gg2013.json.zip", "gg2013.json"]
    in_path = next((p for p in input_candidates if __import__("os").path.exists(p)), None)
    if in_path is None:
        print("WARNING: raw file not found. Put gg2013.json.zip or gg2013.json next to gg_api.py")
        return

    # regex + dash normalization (keep hyphens, standardize all dash variants to '-')
    url_re = re.compile(r"https?://\S+")
    mention_re = re.compile(r"@\w+")
    hashtag_re = re.compile(r"#\w+")
    dash_map = dict.fromkeys(map(ord, "–—‐−‒"), ord("-"))

    def _clean(text: str) -> str:
        if not text:
            return ""
        text = fix_text(text)
        text = _unidecode.unidecode(text)
        text = text.translate(dash_map)                 # keep hyphens
        text = url_re.sub("", text)
        text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)
        text = mention_re.sub("", text)
        text = hashtag_re.sub("", text)
        # remove punctuation EXCEPT hyphen-minus
        text = re.sub(r"[^A-Za-z0-9\s\-]", " ", text)
        text = " ".join(text.split())
        return text

    def _iter_records():
        if in_path.endswith(".zip"):
            with zipfile.ZipFile(in_path) as zf:
                inner = zf.namelist()[0]  # assume single JSON member
                with zf.open(inner, "r") as fh:
                    yield from json.load(io.TextIOWrapper(fh, encoding="utf-8"))
        else:
            with open(in_path, "r", encoding="utf-8") as fh:
                yield from json.load(fh)

    wrote = 0
    with open("tweets_cleaned.jsonl", "w", encoding="utf-8") as out:
        for t in _iter_records():
            raw_text = t.get("text") or ""
            cleaned = _clean(raw_text)

            ts_ms = t.get("timestamp_ms")
            try:
                ts_iso = datetime.fromtimestamp(int(ts_ms) / 1000.0).isoformat() if ts_ms else None
            except Exception:
                ts_iso = None

            rec = {
                "id": t.get("id"),
                "timestamp": ts_iso,
                "screen_name": (t.get("user") or {}).get("screen_name"),
                "user_id": (t.get("user") or {}).get("id"),
                "text": cleaned,           # cleaned, hyphen-preserved text for extraction
                "text_original": raw_text  # optional: for debugging
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Pre-ceremony: wrote {wrote} cleaned tweets to tweets_cleaned.jsonl")
    print("Pre-ceremony processing complete.")
    return

from nlp_pipeline.extract_nominees import extract_nominees
import json

def main():
    '''Main function that orchestrates the Golden Globes analysis.
    
    This function should:
    - Call pre_ceremony() to set up the environment
    - Run the main analysis pipeline
    - Generate and save results in the required JSON format
    - Print progress messages and final results
    
    Usage:
        - Command line: python gg_api.py
        - Python interpreter: import gg_api; gg_api.main()
    
    This is the second function the TA will run when grading.
    
    Note:
        - Do NOT change the name of this function or what it returns
        - This function should coordinate all the analysis steps
        - Make sure to handle errors gracefully
    '''
    # Your code here
    # Load tweets
    # set up environment, extract cleaned tweets
    print("Starting main")
    #pre_ceremony()
    cleaned_path = "tweets_cleaned.jsonl"
    print("Finding Hosts:")
    hosts= get_hosts(YEAR)
    print("Host(s): ", hosts)
    
    print("Analyzing Red Carpet:")
    from red_carpet import find_best_worst
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer


    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("Downloading VADER sentiment lexicon...")
        nltk.download("vader_lexicon")

    rc_results = find_best_worst(cleaned_path, year = YEAR, top_k=5)
    
    print("Analyzing Humor:")
    from humor import find_jokes
    humor = find_jokes(cleaned_path, top_k_people=5, top_k_themes=5)
    
    
    
    # Hardcoded award names (global from gg_api.py)
    global AWARD_NAMES

    # # Run nominee extraction
    # # nominees = extract_nominees(tweets, AWARD_NAMES)

    # # Construct output
    # output = {
    #     "Host": ["Tina Fey", "Amy Poehler"]  # placeholder
    # }

    # for award in AWARD_NAMES:
    #     output[award] = {
    #         "Presenters": [],
    #         "Nominees": nominees.get(award, []),
    #         "Winner": ""
    #     }

    # # Save to final_output.json
    # with open("final_output.json", "w") as f:
    #     json.dump(output, f, indent=2)

    # print("Done, nominees written to final_output.json")

    # print(f"Loaded {len(tweets)} tweets")
    # print("Sample tweet:", tweets[0])

if __name__ == '__main__':
    main()
