import pandas as pd
import re
import zeyrek
from nltk.corpus import stopwords

# nltk.download('stopwords')


DF = pd.read_csv("balanced_data.csv")
output = "lemmatized.csv"
volume = 5000

sentences = DF.iloc[:volume, 0]


lemmatized_sentences = []

"""
def lemmatize_all(sentences, lemmatized_sentences):
    length = sentences.shape[0]

    for i in range(length):
        lemmatized_sentences.append(lemmatize_sentence(sentences[i]))
        print(f"{i} / {volume}")
"""


def lemmatize_all(sentences, temp_size):
    length = len(sentences)
    lemmatized_sentences = []

    for i in range(0, length, temp_size):
        temp_sentences = sentences[i : i + temp_size]
        temp_lemmatized_sentences = [
            lemmatize_sentence(sentence) for sentence in temp_sentences
        ]

        temp_df = pd.DataFrame({"lemmatized_sentence": temp_lemmatized_sentences})
        with open(output, "a", newline="", encoding="utf-8") as file:
            temp_df.to_csv(file, index=False, header=not bool(i))

        lemmatized_sentences.clear()

        print(f"{i} / {length}")

    # Son kalan veriyi CSV dosyasına ekleyin
    if lemmatized_sentences:
        temp_df = pd.DataFrame({"lemmatized_sentence": lemmatized_sentences})
        with open("output.csv", "a", newline="", encoding="utf-8") as file:
            temp_df.to_csv(file, index=False, header=not bool(length // temp_size))


analyzer = zeyrek.MorphAnalyzer()


def lemmatize_sentence(sentence):
    sentence = re.sub("[^a-zA-ZçÇğĞıİöÖşŞüÜ]", " ", sentence)
    sentence = sentence.split()

    text = []

    for word in sentence:

        if not word in set(stopwords.words("turkish")):
            lemmatized_word = analyzer.lemmatize(word)[0][1][0].lower()
            analysis = analyzer.analyze(word)

            if "Neg" in analysis[0][0].morphemes:
                lemmatized_word = lemmatized_word + "negative"

            text.append(lemmatized_word)

    return " ".join(text)


def set_Y(lenght):
    return DF.iloc[:lenght, 1]


if __name__ == "__main__":

    lemmatize_all(sentences, temp_size=1000)
