import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # copied over from ipynb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 12345

if __name__ == '__main__':
    data_dir = 'data_reviews'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N,n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # Print out the first five rows and last five rows
    tr_text_list = x_train_df['text'].values.tolist()
    rows = np.arange(0, 5)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

    print("...")
    rows = np.arange(N - 5, N)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

    # Splitting up training data into training set and validation set
    x_train, x_va, y_train, y_va = train_test_split(x_train_df['text'], y_train_df, test_size=0.2, random_state=SEED)
    y_train = y_train.values.ravel()
    y_va = y_va.values.ravel()

    pipeline = Pipeline(
            steps=[
            ('vectorizer', CountVectorizer(stop_words='english')), # transformer
            ('log_regr', LogisticRegression()), # classifier
            ])

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_va)

    print("Accuracy:", accuracy_score(y_va, predictions))