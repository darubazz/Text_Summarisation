from text_preprocessing import custom_tokenize, language_detection

#from razdel import tokenize
from razdel import sentenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from pathlib import Path


def custom_sentenize(text: str) -> list:
    sents = []
    for line in text.splitlines():
        line = line.lower() if line.strip().endswith("\n") else line
        sents += [sent.text for sent in sentenize(line) if sent.text != ""]
    return sents


class LuhnSummarizer:
    def __init__(self, is_stemmed: bool = True):
        self.is_stemmed = is_stemmed
        self.is_stemmed = True
        self.sf_word_threshold = 0.001
        self.lang = 'RUSSIAN'

    def tokenize_sent(self, sentences: list) -> list:

        tokens = [custom_tokenize(sent) for sent in sentences]

        return tokens

    def create_word_freq_dict(self, text: str) -> dict:  # Bag of Words

        tokens = custom_tokenize(text, self.lang)
        tokens = [x for x in tokens]
        vectorizer = TfidfVectorizer(use_idf=False)
        X = vectorizer.fit_transform(tokens)
        features_names_out = vectorizer.get_feature_names_out()

        freq_df = pd.DataFrame(X.toarray(), columns=features_names_out)  # составляем датафрейм частотностей слов
        # print(f"Features Names Out: {features_names_out}")

        freq_dict = {word: np.sum(freq_df[word].values) / len(list(freq_df.keys())) for word in features_names_out}

        freq_dict = dict(sorted(freq_dict.items(), key=lambda y: y[1],
                                reverse=True))  # сортируем словарь частотностей слов по убыванию
        freq_dict = {k: v for k, v in freq_dict.items() if v >= self.sf_word_threshold}

        return freq_dict

    def get_sentence_significance_word_mask(self, sentence_words_mask: list) -> list:
        first_sf_word_indx = sentence_words_mask.index(1)
        last_sf_word_indx = len(sentence_words_mask) - 1 - sentence_words_mask[::-1].index(1)

        return sentence_words_mask[first_sf_word_indx: last_sf_word_indx + 1]

    def compute_significance_factor(self, freq_dict: dict, sentence: list) -> np.float16:
        sentence_words_mask = [1 if freq_dict[word] else 0 \
                               for word in sentence if word in freq_dict.keys()]
        # print(sentence_words_mask)

        if sum(sentence_words_mask) == 0:
            return 0

        sentence_words_mask = self.get_sentence_significance_word_mask(sentence_words_mask)
        number_of_sf_words = sum(sentence_words_mask)
        total_number_of_bracketed_words = len(sentence_words_mask)

        significance_factor = number_of_sf_words ** 2 / total_number_of_bracketed_words

        return significance_factor

    def summarize(self, text: str) -> str:
        self.lang = language_detection(text)
        sentences = custom_sentenize(text)
        text_freq_dict = self.create_word_freq_dict(text)

        sentences_sf = []
        for sent in sentences:
            sentence_tokens = custom_tokenize(sent, self.lang)
            sentences_sf.append(
                self.compute_significance_factor(text_freq_dict, sentence_tokens) if len(sentence_tokens) > 0 else 0)

        sentences_sf.sort(reverse=True)
        sentence_sf_threshold_percentile_75 = np.percentile(sentences_sf, 75)

        summary_sentences = [sent for i, sent in enumerate(sentences) if
                             sentences_sf[i] > sentence_sf_threshold_percentile_75]

        return "".join(summary_sentences)


if __name__ =="__main__":
    summarizator = LuhnSummarizer()

    text = ""
    with open (Path("data/article_ru.txt"), "r", encoding='utf-8') as f:
        text = "".join(f.readlines())
    print(summarizator.summarize(text))
