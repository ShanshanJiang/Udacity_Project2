import nltk

nltk.download(
    [
        "punkt",
        "stopwords",
        "words",
        "averaged_perceptron_tagger",
        "maxent_ne_chunker",
        "wordnet",
    ]
)

import sys
import re
import nltk
import pickle
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    This function is to load the data from database file path

    Args:
        database_filepath (string): the file path of the database

    Returns:
        X (pandas dataset): independent variables
        Y (pandas dataset): dependent variables
        category_names (list of string): category names of the dependent variables
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("DisasterResponse", con=engine)
    df.dropna(axis=0, subset=df.columns[4:], inplace=True)
    X = df["message"]  # Message Column
    Y = df.iloc[:, 4:]  # Classification label
    Y = Y.loc[:, (Y != 0).any(axis=0)]
    Y = Y.loc[:, (Y != 1).any(axis=0)]  # drop any column with all 0 or all 1
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    This function is to split the text into tokens

    Args:
        text (string): message sentence

    Returns:
        Lemmed (list): a list of Lemmatized token
    """
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenization
    words = word_tokenize(text)

    # remove stop words
    words_no_stop = [w for w in words if w not in stopwords.words("english")]

    # Lemmatization
    Lemmed = [WordNetLemmatizer().lemmatize(w) for w in words_no_stop]

    return Lemmed


def build_model():
    """
    This function is to build a classifier model

    Returns:
        model
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(estimator=RandomForestClassifier())),
        ]
    )

    parameters = {
        "tfidf__use_idf": (True, False),
        "clf__estimator__n_estimators": [50, 100],
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function is to evaluate the model built in the build_model function

    Args:
        model : model developed in the build_model function
        X_test (pandas database): test set of independent variable
        Y_test (pandas database): test set of dependent variable
        category_names (list of string): list of category names

    Returns:
        None
    """
    Y_pred = model.predict(X_test)

    i = 0
    for col in Y_test:
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i += 1
    accuracy = (Y_pred == Y_test.values).mean()
    print("The model accuracy is {:.3f}".format(accuracy))
    return


def save_model(model, model_filepath):
    """
    This function is to save the model

    Args:
        model : selected model
        model_filepath (string): the destination to save the model
    """
    file_name = model_filepath
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
