
# ------------- needed packajes & functions for first Phase - Sentiment Analysis --------


#from sklearn.preprocessing import StandardScaler
#from numpy import load as np_load
#from numpy import save
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# from transformers import pipeline
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string   # we need it for Punctuations removal
# or we can use from nltk.corpus import stopwords
import re
import pickle
from joblib import load
# roberta intialization
# roberta_pl = pipeline("sentiment-analysis", max_length=512, truncation=True)
vader = pickle.load(open('pkl_files/vader_sentiment_model.pkl', 'rb'))
roberta = pickle.load(open('pkl_files/roberta_sentiment_model.pkl', 'rb'))

# define user text to make it globally to use it in sentiment analysis functions
user_text = ''

# clean the text for sentiment


def sentiment_clean_text(text):
    '''
    this function take text and clean it  

    Parameters
    ----------
    text : string 

    Returns
    -------
    text_list : string after cleaning.

    '''

    # a. first turn letters into lowercase
    text = text.lower()

    # b. second removes unalfabetic signs
    text = re.sub("[^a-zA-Z]", " ", text)

    # c. third remove all Punctuations.
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text

# User Sentiment Fuction


def calc_user_sentiment_score(answers):
    """
    takes: list of answers 
    return: dictionary with user positive and negative scores 
    """
    user_text = ''

    # First Read Screening DataFrame
    df = pd.read_csv('data/depression_screening.csv')

    # intialize needed variables
    pos_score = 0
    neg_score = 0
    choice = 0

    # loop over answers to calc scores
    for index, ans in enumerate(answers):
        choice = ans

        if choice == 1:
            pos_score += 1
            user_text += ' ' + df.never[index]

        elif choice == 2:
            neg_score += 0.33
            user_text += ' ' + df.sometimes[index]

        elif choice == 3:
            neg_score += 0.67
            user_text += ' ' + df.usually[index]

        else:
            neg_score += 1
            user_text += ' ' + df.always[index]

    return {"Positive": pos_score, "Negative": neg_score}, 1 if pos_score > neg_score else 0, user_text


# Vader Sentiment Function
def calc_vader_sentiment_score(text):

    # return dictionary of scores

    scores_dict = vader.polarity_scores(text)
    score = 0

    if (scores_dict['neg'] < scores_dict['pos']):
        score += 1

    return scores_dict, score


# roberta Sentiment Function
def calc_roberta_sentiment_score(text):

    # return dictionary of scores
    scores_dict = roberta(text)[0]
    score = 0

    if (scores_dict['label'] == 'POSITIVE'):
        score += 1

    return scores_dict, score


# ------------- needed libraries & functions for Second Phase - Depression Detection --------

# Input other libraries and necessary files
'''

# define needed objects
texts= []
nb =  GaussianNB()
svm = SVC(kernel='linear')  # poly -> 72 , sigmoid -> 83
lr = LogisticRegression()


# train Model

vectorizer = load(open("vectorizer.pkl", "rb"))

scaler = load(open("scaler.pkl", "rb"))

def train_depression_data(df):

    for i in df.clean_text:
        
        texts.append(i)
    
        
    # it removes unnecesarry words and finds most used 550 words
    #vectorizer = CountVectorizer(stop_words="english", max_features=550)
    # Save the scaler object to a file
    #with open('vectorizer.pkl', 'wb') as f:
    #    pickle.dump(vectorizer, f)
    
    # Create a vectorizer object and vectorize the texts
    vectorized_texts = vectorizer.fit_transform(texts)

    # Create a scaler object and scale the vectorized texts
    #scaler = StandardScaler()
    scaled_texts = scaler.fit_transform(vectorized_texts.toarray())

    # Save the scaler object to a file
    #with open('scaler.pkl', 'wb') as f:
    #    pickle.dump(scaler, f)


    # Set X and Y
    X = scaled_texts
    y = df["is_depression"].values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # fit the With Naive-Bayes
    nb.fit(X_train, y_train)
    # Save the scaler object to a file
    with open('is_depressed_nb.pkl', 'wb') as f:
        pickle.dump(nb, f)


    # Train the SVM classifier
    svm.fit(X_train, y_train)
    # Save the scaler object to a file
    with open('is_depressed_svm.pkl', 'wb') as f:
        pickle.dump(svm, f)


    # Train the logistic regression classifier
    lr.fit(X_train, y_train)
    # Save the scaler object to a file
    with open('is_depressed_lr.pkl', 'wb') as f:
        pickle.dump(lr, f)


# read  file
df = pd.read_csv('data/is_depression.csv')

# call training function 
train_depression_data(df)
'''

nb = load(open('pkl_files/is_depressed_nb.pkl', 'rb'))
svm = load(open('pkl_files/is_depressed_svm.pkl', 'rb'))   # poly -> 72 , sigmoid -> 83
lr = load(open('pkl_files/is_depressed_lr.pkl', 'rb'))

with open('pkl_files/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


with open('pkl_files/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# is_depressed function - for prediction
def is_depressed(text):
    texts = [] 

    # Preprocess and scale a set of text
    texts.insert(0, text)
    df = pd.read_csv('data/is_depression.csv')

    for i in df.clean_text:
        texts.append(i)
    vectorized_new_text = vectorizer.fit_transform(texts)
    text = scaler.fit_transform(vectorized_new_text.toarray())

    # prediction on text
    # count = CountVectorizer(stop_words = "english", max_features = 550) #it removes unnecesarry words and finds most using 550 words
    # user_text = vectorizer.fit_transform([text]).toarray()

    # is depressed using Naive Bayse
    is_depressed_NB = nb.predict([text[0]])

    # Predict the classes of the validation set
    is_depressed_SVM = svm.predict([text[0]])

    # Predict the classes of the validation set
    is_depressed_LR = lr.predict([text[0]])

    return 1 if (is_depressed_NB + is_depressed_LR + is_depressed_SVM) >= 2 else 0

# ------------- needed libraries & functions for third Phase - Red Flags Detection --------


def red_flags_detect(answers, text):

    # 10 Most Common Red Flags Emotions
    red_flags_emotions = ['lost', 'scared', 'afraid', 'hurt', 'alone',
                        'lonely', 'suffering', 'depressed', 'rejected', 'terrified']

    # check for red flags
    red_flag = False

    for word in text.split():
        if word in red_flags_emotions:
            red_flag = True
            break

    if answers[5] > 2 or answers[8] > 2 or red_flag == True:

        return 1

    else:

        return 0


# Excuted function for flask - flask Function :: Important

def excute_detection_model(answers, text):
    '''
    this fuction takes
    answers: list of user answers [1->'Never',2->'Sometimes',3->'Usually',4->'Always']
    text : user text 

    '''
    # 1. Define Needed Variables
    user_score_dict = {}
    user_score = 0

    vader_score_dict = {}
    vader_score = 0

    roberta_score_dict = {}
    roberta_score = 0

    sentiment_result = 'Positive'
    depression_test_result = 'No Depression'
    red_flags = "No Red Flags"
    
    # 2. Concatenate text and clean it  for sentiment analysis
    text = sentiment_clean_text(text)

    # 3. Do sentiment analysis (User - Vader - Roberta)
    user_score_dict, user_score, user_text = calc_user_sentiment_score(answers)

    # assing new generated text for next phases
    final_text = user_text + ' ' + text

    vader_score_dict, vader_score = calc_vader_sentiment_score(final_text)
    roberta_score_dict,roberta_score = calc_roberta_sentiment_score(final_text)

    # condition meaning: two sentiment test gave us negative result so we will continue the analysis
    if (user_score + vader_score + roberta_score < 2):

        # change value for sentiment
        sentiment_result = "Negative"

        # continue to the next phase - Depression Detection
        if is_depressed(final_text) == 1:
            depression_test_result = "Depressed"

            # continue to the next phase - Red Flags Detection
            if red_flags_detect(answers, final_text) == 1:

                red_flags = "Detected Red Flag"
                

    return sentiment_result, depression_test_result, red_flags


