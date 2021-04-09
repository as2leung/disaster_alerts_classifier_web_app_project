# Disaster Response Pipeline Project


## Table of Contents
* [Purpose](#purpose)
* [Running the web app](#running)
* [Running all pipelines and web app](#full)
* [Web App Output](#output)
* [List of key files](#list_of_key_files)
* [Python Libraries](#python_libraries)
* [Download Model - Pickle](#pickle)
* [Note on imbalanced classes](#note)
* [Creator & Credits](#creators)



## Purpose

This project is a web app API that classifies disaster messages to a category. Data from Figure Eight is used to calibrate the underlying model (Multi-Output Classifier with Random Forest Classifier).
There are three underlying components to the web app:

1.	ETL pipeline Python script
2.	ML pipeline Python script
3.	Flask Web App with two webpages

NLTK is used to carry out the required text extraction and normalization on the disaster messages. Messages are then converted from raw text to a matrix of TF-IDF features to predict the 36 disaster message categories. A custom transformer was also created to identify whether the first word of the message is a verb to include as another feature.
The scikit-learn pipeline and GridSearchCv are used to select the appropriate hyperparameters for the final model.



## Running just the web app <a name="running"></a>

To run just the web app, first download the pickle file (see Download Model - Pickle below), rename the pickle to *classifier.pkl"* and then place the pickle file into the **model** folder.

Next, run the "run.py" file in the app folder and then go to http://127.0.0.1:3001/ on your web browser



## Running all pipelines and web app <a name="full"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        *`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_alerts.db`*

    - To run ML pipeline that trains classifier and saves
        *`python models/train_classifier.py data/diaster_alerts.db models/classifier.pkl`*

2. Run the following command in the app folder to run your web app.
    *`python run.py`*

3. Go to http://127.0.0.1:3001/

## Web App Output <a name="output"></a>

The following are screenshots of the actual Web App. If the app is run correctly, it should look as follows:

### Web App Homepage

![Homepage](https://github.com/as2leung/disaster_alerts_classifier_web_app_project/blob/main/screenshots/00_homepage.PNG)

### Web App Classifier
![User Input Page](https://github.com/as2leung/disaster_alerts_classifier_web_app_project/blob/main/screenshots/01_categorize_user_input.PNG)

## File structure and list of key files <a name="list_of_key_files"></a>

* **app folder**
	* template folder
		* *master.html*  # main page of web app - displays three visualizations
		* *go.html*  # classification result page of web app
	* run.py  # Flask file that runs app (change host ip address here)

* **data**
	* *disaster_categories.csv*  # data to process 
	* *disaster_messages.csv*  # data to process
	* *process_data.py*		   #ETL pipeline script		
	* *InsertDatabaseName.db*   # database to save clean data to

* **models**
	* *train_classifier.py* #ML pipeline script
	* *classifier.pkl*  # saved model 

* **README.md**



## Python_Libraries

* nltk
* sklearn
* joblib
* sqlqlchemy
* time
* numpy
* re
* pandas

## Download Model Pickle (large file) <a name="pickle"></a>

Since the model pickle is very large in size, I have uploaded elsewhere. You can download them here. Please rename the pickle to *classifier.pkl"* and then place the pickle file into the **model** folder and follow the run steps above.

[Pickle File Download](https://drive.google.com/drive/folders/1Z7OuyjNlF5WFYrFphaP18d-TeQEb6uPX?usp=sharing)


## Note on imbalanced classes <a name="note"></a>

For most of the 36 classification categories the categories suffer from imbalanced classes, where there is a majority of observations in one class. The issue with imbalanced classes is that it results in the classifier model usually predicting the majority class the most of the time.

In terms of ways to combat this issue, resampling methods can be used to help balance the classes in a statistically sound way or the use of ensemble methods where the classifier calibrates multiple models/learners to find a more robust model. The random forest classifier used in this web app is one such ensemble method.

## Creators & Credits <a name="creators"></a>

* Andrew Leung
    - [https://github.com/as2leung](https://github.com/as2leung)
* Udacity
	 - [Website](https://www.udacity.com/)

