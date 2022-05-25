from nltk.corpus import stopwords
from string import punctuation
from razdel import tokenize
from nltk.stem.snowball import SnowballStemmer

# Create stopwords list
russian_stopwords = stopwords.words("russian")


def custom_tokenize(text: str) -> list:
    text = text.lower()
    tokens = list(tokenize(text))
    tokens = [token.text for token in tokens if token.text not in russian_stopwords \
              and token.text != " " \
              and token.text.strip() not in punctuation]
    stemmer = SnowballStemmer(language='russian')
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens
