Tweet Mining and Golden Globes: Ambar Luzio, Yung-ching Lai, Alan Wang

Tested with Python 3.10
Data:
- must have gg2013.json or gg2013.json.zip in directory (or rename data to one of these)

Environment Setup:
- activate a virtual environment and install packages
- must have pip installed/upgraded (% python -m pip install)
- % pip install -r requirements.txt

Required Packages:
- same contents as requirements.txt
spacy
nltk
jupyterlab
ftfy
unidecode
inflection
langdetect
icrawler
rapidfuzz
cinemagoer

Downloads:
- NLTK VADER lexicon used for sentiment analysis. The code will download it automatically on the first run.
- Can pre-download it with: python -c "import nltk; nltk.download('vader_lexicon')"

To Run:
- % python gg_api.py
- This will run main() that calls the following in order
    - pre_ceremony()
    - get_hosts()
    - get_awards()
    - get_nominees()
    - get_winner()
    - get_presenters()
- after these it will make calls for additional goals
    - red carpet: finds who were the worst/best dressed and uploads a picture of each
        - Results: Bradley Cooper was Best Dressed and Sofia Vergara was Worst Dressed 
    - humor: Finds the top funniest people of the night and popular jokes themes
        - Funniest People: ['Amy Poehler', 'Tina Fey', 'Will Ferrell', 'James Cameron', 'Kristen Wiig']
    - 


Github repo:
https://github.com/luzii24/NLP_337_HW1

