# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import time

import numpy as np
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')



def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_categorized',engine)
    X = df["message"]
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
        
    def starting_verb(self,text):
        sentence_list = nltk.sent_tokenize(text)
        #print(sentence_list)
        for sentence in sentence_list:
            #print(sentence)
            pos_tags = nltk.pos_tag(tokenize(sentence))
            #return pos_tags
            if not pos_tags:
                return False
            else:
                first_word, first_tag = pos_tags[0]

                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        X_tagged = X_tagged.fillna(False)
        return pd.DataFrame(X_tagged)

def build_model():
    pipeline3 = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('starting_verb', StartingVerbExtractor())
    ])),

    ('clf',  MultiOutputClassifier(RandomForestClassifier(max_features='log2')))
    ])
    
    parameters ={
        'clf__estimator__min_samples_split':[2,4,6,8],
        'clf__estimator__max_features':['auto', 'sqrt', 'log2']

    }
    
    
    cv = GridSearchCV(pipeline3, param_grid=parameters)
    
    return cv
    
   
def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    for i in range(Y_pred.shape[1]):
        labels = np.unique(Y_pred[i]).tolist()
        targets = [str(i) for i in labels]
        print("model {}".format(category_names[i]))
        print(classification_report(Y_test.iloc[:,i],Y_pred[:,i], labels=labels, target_names=targets))

    
    


def save_model(model, model_filepath):
    joblib_file = model_filepath
    joblib.dump(model, joblib_file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()