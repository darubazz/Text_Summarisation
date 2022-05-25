from nltk.corpus import stopwords
from string import punctuation
from razdel import tokenize
from nltk.stem.snowball import SnowballStemmer
import pycld2 as cld2

# Create stopwords dictionary
stopwords_dict = {'RUSSIAN': stopwords.words("russian"),
                  'ENGLISH': stopwords.words("english")}


def language_detection(text: str) -> str:
    return cld2.detect(text)[2][0][0]


def custom_tokenize(text: str) -> list:
    stopwords = language_detection(text)
    text = text.lower()
    tokens = list(tokenize(text))
    tokens = [token.text for token in tokens if token.text not in stopwords \
              and token.text != " " \
              and token.text.strip() not in punctuation]
    stemmer = SnowballStemmer(language=stopwords.lower())
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens
