from sklearn.feature_extraction.text import CountVectorizer
import pickle
from read_from_lemmatized import read_lemmantized
from preprocess import set_Y
import pandas as pd

if __name__ == "__main__":

    source = "lemmatized.csv"

    lemmatized_sentences = read_lemmantized(source)

    cv = CountVectorizer(max_features=2000)

    ## Ever you update your dataset. it's enough that execute these code once .

    for i in range(len(lemmatized_sentences) - 1, 0, -1):

        try:
            X = cv.fit_transform([lemmatized_sentences[i]]).toarray()
            print(i)
        except:
            print(f"{i}. index is not transformable.")
            lemmatized_sentences.pop(i)

    pd.DataFrame(lemmatized_sentences).to_csv(source, index=False)

    ##

    X = cv.fit_transform(lemmatized_sentences).toarray()
    Y = set_Y(len(lemmatized_sentences))

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=104, test_size=0.33, shuffle=True
    )

    from sklearn.naive_bayes import MultinomialNB

    mnb = MultinomialNB(alpha=1.2123)
    mnb.fit(X_train, y_train)

    y_pred = mnb.predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    with open("model.pkl", "wb") as model_file:
        pickle.dump(mnb, model_file)

    with open("countVectorizer.pkl", "wb") as model_file:
        pickle.dump(cv, model_file)
