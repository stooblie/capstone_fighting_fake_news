'''
Functions for predicting the fake images from the comparison results.
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import GridSearchCV

def rf_model(df, estimators, features, depth):
    '''
    Random Forest model.

    :param df: dataframe of comparison results.
    :param n_estimators: trees to use in the random forest.
    :param max_features: features to select at each split in the random forest.
    :param max_depth: max depth of teh random forest trees.
    :return rf_pred, rf_recall, rf_precision: the prediction results, recall score, and precision score.
    '''

    X_train, X_test, y_train, y_test = train_test_split(list(zip(df['histogram'].values, df['ssim'].values, df['tweet_match'].values, df['gradient_similarity'].values)), df['matches'].astype(int).values)

    rf = RandomForestClassifier(n_estimators=estimators, max_features=features, max_depth=depth)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_recall = recall_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)

    return rf_pred, rf_recall, rf_precision, rf.feature_importances_
