# kaggle_quora_competition

These files are the summary of our (frucci, aborgher) submission on the Quora Kaggle competition (https://www.kaggle.com/c/quora-question-pairs).

The goal of the competition was to predict duplicate questions (question with the same meaning).

We joined the competition to learn & have fun while deadline was 1 month to go.
Things tried: xgboost, LSTM, GRU and some libraries used for NLP in python (gensim, nltk, treetagger)

We avoided the usage of features which cannot be created and used in a real-situation (where the test is really unknown) and so we didn't achieve the best score possible on the leaderboard.

Our final score was about 0.32 logloss on private leaderboard achieved with the LSTM neural network (top 35% on ~3400).

# Legend:
- Quora_duplicate.ipynb: main jupyter-notebook used for features extraction and to run the model
- quoradefs.py: many defined functions used in Quora_duplicate 
- Tagger.ipynb: add verb-nouns-etc.. composition to the phrases and generate some csv to be used in Quora_duplicate
- Simple_LSTM.ipynb/run_LSTM.py: code to train a LSTM using keras and tensorflow
- run_LSTM.sh: bash file to run many neural networks

- get_phrase_correction.py: using pyenchant to check how are bad written the questions in train and test
- Backup_code_not_used.ipynb

