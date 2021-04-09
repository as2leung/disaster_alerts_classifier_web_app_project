import sys
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
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
import numpy as np


import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

def tokenize(text):
    """ Custom tokenizer for pickle to work"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to return bool feature for whether a text 
    string starts with a verb

    """
    
        
    def starting_verb(self,text):
        
        """ method to extract POS from first word in text string

        """
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
        """Apply custom transformer to pd.Series"""
        X_tagged = pd.Series(X).apply(self.starting_verb)
        X_tagged = X_tagged.fillna(False)
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///../data/disaster_alerts.db')
df = pd.read_sql_table('messages_categorized', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    values_per = df['genre'].value_counts(normalize= True).round(3).multiply(100).tolist()
    labels_per_name = df['genre'].value_counts().index.tolist()
    
    color=np.array(['rgb(55, 83, 109)']*df.iloc[:,4:].shape[0])
    color2=np.array(['rgb(255, 40, 109)']*df.iloc[:,4:].shape[0])
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
         {
            'data': [
                Bar(
                    x=list(df.iloc[:,4:].columns),
                    y=df.iloc[:,4:].sum(),
                    marker=dict(color=color.tolist())
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Reponse/Outcome',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Reponse/Output Variables"
                }
            }
        },
            {
                'data': [
                   
                      Pie(labels=labels_per_name, values=values_per, textinfo='label+percent'
                            )
                ],

                'layout': {
                    'title': 'Percentage Distribution of Message Genres',
                    

                }
            }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()