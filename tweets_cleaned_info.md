## tweets_cleaned.jsonl format
- data size: 125185
- columns:  
`['timestamp', 'screen_name', 'user_id', 'text', 'hashtags', 'is_retweet', 'is_quote', 'original_author', 'original_text', 'user_comment']`
- example (normal tweet):  
```json
{
    "timestamp": "2013-01-13T18:45:38",
    "screen_name": "theAmberShow",
    "user_id": 14648726,
    "text": "what s making sofia vergara s boobs stay like that magic witchcraft",
    "hashtags": [
        "GoldenGlobes"
    ],
    "is_retweet": false,
    "is_quote": false,
    "original_author": null,
    "original_text": "",
    "user_comment": ""
}
```
- example (rewteet):
```json
{
    "timestamp": "2013-01-13T18:45:38",
    "screen_name": "SweetyPW",
    "user_id": 35498686,
    "text": "kerry washington is everything dying over her miu miu gown",
    "hashtags": [
        "goldenglobes"
    ],
    "is_retweet": true,
    "is_quote": false,
    "original_author": "FabSugar",
    "original_text": "kerry washington is everything dying over her miu miu gown",
    "user_comment": ""
}
```

- Example use of the data:
```python
from collections import Counter
import pandas as pd

file = "tweets_cleaned.jsonl"
df = pd.read_json(file, lines=True)
hashtags = Counter(tag for tags in df['hashtags'] for tag in tags)
print(hashtags.most_common(20))
# output:
# [('GoldenGlobes', 76228), ('goldenglobes', 25195), ('GetGlue', 684), ('Argo', 654), ('GoldenGlobe', 513), ('Homeland', 472), ('JodieFoster', 425), ('Goldenglobes', 379), ('redcarpet', 370), ('Girls', 352), ('LesMis', 275), ('Lincoln', 270), ('GIRLS', 263), ('Skyfall', 244), ('GOLDENGLOBES', 229), ('RedCarpet', 222), ('JenniferLawrence', 219), ('LesMiserables', 206), ('TinaFey', 197), ('homeland', 197)]
```
