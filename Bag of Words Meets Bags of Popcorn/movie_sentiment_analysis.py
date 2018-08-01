# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import nltk
import os
import re
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_dataset(file):
    """
        load data based on specific file
    """
    rawdata = {
        'unlabeled_train_data': 'unlabeledTrainData.tsv',
        'labeled_train_data': 'labeledTrainData.tsv',
        'test_data': 'testData.tsv'}
    if file not in rawdata:
        print('No such a file')
        return
    path = os.path.join('.', 'data', rawdata[file])  # folder "data" to store the raw data
    try:
        dataframe = pd.read_csv(path, sep='\t', escapechar='\\')
    except IOError:
        print("Error: No such directory or file...")
    else:
        print("Data loaded successfully...")
    return dataframe


def clean_text(text):
    """
        clean the HTML data
    """
    parse_text = BeautifulSoup(text, 'html.parser').get_text()
    word_text = re.sub(r'[^a-zA-Z]', ' ', parse_text)   # only train model with words
    words = word_text.lower().split()
    return words


def tokenize_sentences(review):
    review_sentences = tokenizer.tokenize(review.strip())  # review are tokenized into sentences
    cleaned_sentences = [clean_text(sen) for sen in review_sentences if sen]  # clean data for each sentence
    return cleaned_sentences


def train_vector_model(dimension=300, num_word=40, num_workers=2, window_size=10, sampling=1e-3):
    """
    input: parameters to train vector model
        dimension: Word vector dimensionality
        num_word: Minimum word count
        num_workers: Number of threads to run in parallel
        window_size: The window size
        sampling: Downsample setting for frequent words
    """
    model_name = '{}dimension_{}minwords_{}window_size.model'.format(dimension, num_word, window_size)
    print('Training model...')
    model = Word2Vec(sentences, workers=num_workers, size=dimension, min_count=num_word,
                     window=window_size, sample=sampling)

    try:
        model.save(os.path.join('.', 'models', model_name))  # model is stored in folder "models"
    except IOError:
        print("Error: No such directory")
    else:
        print("Model saved successfully")


def load_model(use_trained_model=False):
    if not use_trained_model:
        try:
            model = Word2Vec.load(os.path.join('.', 'models', 'all_fin_model_lower'))
        except IOError:
            print("Error: No such directory or file")
        else:
            print("Model loaded successfully")

    else:
        try:
            model = Word2Vec.load(os.path.join('.', 'models', '300dimension_40minwords_10window_size.model'))
        except IOError:
            print("Error: No such directory or file")
        else:
            print("Model loaded successfully")

    return model


def to_review_vector(review, model):
    """
        convert the sentences into vectors, average vectors are used
    """
    words = clean_text(review)
    vectors = np.array([model[word] for word in words if word in model])
    return pd.Series(np.mean(vectors, axis=0))


if __name__ == "__main__":
    # use unlabeled_train_data to train a word2vec model
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print('Loading the unlabeled train data...')
    unlabeled_train_data = load_dataset('unlabeled_train_data')
    print('Loading is done...')

    print('Cleaning the reviews...')
    sentences = sum(unlabeled_train_data.review.apply(tokenize_sentences), [])
    print('Cleaning is done...')
    train_vector_model()

    # use labeled_train_data to train random forest model and test on validation dataset
    labeled_train_data = load_dataset('labeled_train_data')
    train, test = train_test_split(labeled_train_data, test_size=0.2, random_state=7)  # train : validation = 4 : 1

    # Step1: use pre-downloaded financial model
    model_financial = load_model()  # first load financial model and check the accuracy
    train_data_financial = train.review.apply(to_review_vector, model=model_financial)
    test_data_financial = test.review.apply(to_review_vector, model=model_financial)

    forest = RandomForestClassifier(n_estimators=200, random_state=7, oob_score=True, n_jobs=-1)
    forest = forest.fit(train_data_financial, train.sentiment)
    print("Results for financial-vector model...", '\n')
    print(classification_report(test.sentiment, forest.predict(test_data_financial)))
    print('\n')

    # Step2: use trained reviews model
    model_trained = load_model(use_trained_model=True)  # load trained reviews model and check its accuracy
    train_data_trainedModel = train.review.apply(to_review_vector, model=model_trained)
    test_data_trainedModel = test.review.apply(to_review_vector, model=model_trained)

    forest = RandomForestClassifier(n_estimators=200, random_state=7, oob_score=True, n_jobs=-1)
    forest = forest.fit(train_data_trainedModel, train.sentiment)
    print("Results for reviews-vector model...", '\n')
    print(classification_report(test.sentiment, forest.predict(test_data_trainedModel)))