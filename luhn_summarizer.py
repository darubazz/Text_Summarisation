import pandas as pd
import numpy as np
from razdel import sentenize

from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path
from text_preprocessing import custom_tokenize, language_detection


def custom_sentenize(text: str) -> list:
    sents = []
    for line in text.splitlines():
        line = line.lower() if line.strip().endswith("\n") else line
        sents += [sent.text for sent in sentenize(line) if sent.text != ""]
    return sents


class LuhnSummarizer:
    def __init__(self):
        self.word_threshold = 0.001
        self.lang = 'RUSSIAN'

    def tokenize_sent(self, sentences: list) -> list:
        tokens = [custom_tokenize(sent) for sent in sentences]
        return tokens

    def create_word_freq_dict(self, text: str) -> dict:  # Bag of Words

        tokens = custom_tokenize(text, self.lang)
        tokens = [x for x in tokens]
        vectorizer = TfidfVectorizer()  # use_idf=False
        X = vectorizer.fit_transform(tokens)
        features_names_out = vectorizer.get_feature_names_out()

        freq_df = pd.DataFrame(X.toarray(), columns=features_names_out)
        freq_dict = {word: np.sum(freq_df[word].values) / len(list(freq_df.keys())) for word in features_names_out}
        freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
        freq_dict = {k: w for k, w in freq_dict.items() if w >= self.word_threshold}

        return freq_dict

    def create_sent_important_word_mask(self, sentence_words_mask: list) -> list:
        first_important_word_index = sentence_words_mask.index(1)
        last_important_word_index = len(sentence_words_mask) - 1 - sentence_words_mask[::-1].index(1)

        return sentence_words_mask[first_important_word_index: last_important_word_index + 1]

    def count_significance_factor(self, freq_dict: dict, sentence: list) -> np.float16:

        sentence_words_mask = [1 if freq_dict[word] else 0 \
                               for word in sentence if word in freq_dict.keys()]
        if sum(sentence_words_mask) == 0:
            return 0

        sentence_words_mask = self.create_sent_important_word_mask(sentence_words_mask)
        number_of_important_words = sum(sentence_words_mask)
        number_of_all_words = len(sentence_words_mask)

        significance_factor = number_of_important_words ** 2 / number_of_all_words

        return significance_factor

    def summarize(self, text: str) -> str:
        self.lang = language_detection(text)
        sentences = custom_sentenize(text)
        text_freq_dict = self.create_word_freq_dict(text)

        sentences_significance = []
        for sent in sentences:
            sentence_tokens = custom_tokenize(sent, self.lang)
            sentences_significance.append(
                self.count_significance_factor(text_freq_dict, sentence_tokens) if len(sentence_tokens) > 0 else 0)

        sentences_significance.sort(reverse=True)
        sentence_threshold = np.percentile(sentences_significance, 75)
        summary_sentences = [sent for i, sent in enumerate(sentences) if
                             sentences_significance[i] > sentence_threshold]

        return "".join(summary_sentences)


if __name__ == "__main__":
    summarizer = LuhnSummarizer()

    text = ""
    with open(Path("data/article_ru.txt"), "r", encoding='utf-8') as f:
        text = "".join(f.readlines())
    print(summarizer.summarize(text))
