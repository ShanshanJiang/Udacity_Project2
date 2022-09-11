# Disaster Response Pipeline Project

### Table of Contents

1. [Summary of the project](#summary)
2. [File Descriptions](#files)
3. [Insturctions](#instructions)


## Summary of the project<a name='summary'></a>

The project to classify the messages during nature disasters. By analyzing real messages sent during natural disasters (Data provided by [Appen](https://appen.com/)) and using machine learning techiniques to create a multi-output supervised learning model to categorize the messages, the result is visually presented on a web app. 

## File Descriptions<a name='files'></a>

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models

|- train_classifier.py
|- classifier.pkl # saved model

README.md


## Instructions<a name='instructions'></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
