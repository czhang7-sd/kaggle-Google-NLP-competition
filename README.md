kaggle competition:  Sentiment analysis for movie reviews using Google's Word2Vec
============================================

Implementation of a Word2vec based sentiment analysis model in Python.

The approach consists of two major steps: movie reviews pre-processing and modeling.

The movie reviews consist of:

1. 'unlabeledTrainData.tsv': movie reviews data without sentiment label.  
2. 'labeledTrainData.tsv':  sentiment labeled movie reviews data.
3. 'testData.tsv': the test data we need to submit as final results. Don't use it locally. Need to submit it to Kaggle website.


The modeling consists of two parts:

Part 1:
1. Split labeled train data into train dataset and validation dataset.
2. Use pre-trained "Financial" Word2vec model. The financial text materials used to train this model are much larger than the whole movie reviews. Vectorized the labeled dataset with this model.
3. Train classifier and test it on the validation set.


Part 2:
1. Use unlabeled movie reviews to train a domain-specific Word2vec model. Compared with the financial one.
2. Split labeled train data into train dataset and validation dataset.
3. Use currently trained "movie reviews" Word2vec model to vectorize the labeled dataset.
4. Train classifier and test it on the validation set.


### Result comparison:
  
Even though the "financial Word2vec" model is trained with far more larger text dataset, the final result still shows the Word2vec trained by movie reviews has a much better classification result on movie reviews sentiment analysis. This means a "domain-specific" processing can largely improve the NLP tasks. 
