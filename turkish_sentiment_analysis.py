import pandas as pd
import re
import threading
import zeyrek
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pickle

DF = pd.read_csv("data.csv")

volume = 100000

sentences = DF.iloc[:volume, 0]


lemmatized_sentences = []


def lemmatize_all(sentences, lemmatized_sentences):
    length = sentences.shape[0]

    for i in range(length):
        lemmatized_sentences.append(lemmatize_sentence(sentences[i]))


analyzer = zeyrek.MorphAnalyzer()


def lemmatize_sentence(sentence):

    sentence = re.sub("[^a-zA-ZçÇğĞıİöÖşŞüÜ]", " ", sentence)
    sentence = sentence.split()
    sentence = [
        analyzer.lemmatize(word)[0][1][0].lower()
        for word in sentence
        if not word in set(stopwords.words("turkish"))
    ]
    return sentence


if __name__ == "__main__":

    lemmatize_all(sentences, lemmatized_sentences)

    cv = CountVectorizer(max_features=3000)

    X = cv.fit_transform(sentences).toarray()
    Y = DF.iloc[:volume, 1]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=104, test_size=0.25, shuffle=True
    )

    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    with open("model.pkl", "wb") as model_file:
        pickle.dump(gnb, model_file)

    with open("countVectorizer.pkl", "wb") as model_file:
        pickle.dump(cv, model_file)
