"""
This code to Know training results for classficiation models to decide what is the best algorithm to use  
"""



# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from transformers import pipeline

user_text = 'usually negative state'
# print (SentimentIntensityAnalyzer().polarity_scores(user_text))
# pipe = pipeline("sentiment-analysis", max_length=512, truncation=True)
# print (pipe(user_text))



# In[]

'''
Text Preprocessing - cleaning 

1. remove unalfabetic signs from string
2. turn all to lowercase  
2. Tokenization turn string to list 
3. remove stop words 
4. lemmatization - reducing a word to it's base 

'''
import pandas as pd 
import re
import nltk
#nltk.download('omw-1.4')
#nltk.download("stopwords")
# read  file      
depression_df = pd.read_csv('data/is_depression.csv')

texts=[]
lemmatize=nltk.WordNetLemmatizer()


for i in depression_df.clean_text:
    
#     #no need for those steps as the text is already cleaned 
#     text=re.sub("[^a-zA-Z]"," ",i) # it removes unalfabetic signs
#     text=nltk.word_tokenize(text,language="english") # it tokenizes our words
#     text=[lemmatize.lemmatize(word) for word in i] # it lemmatizes reducing a word to its base or dictionary form (ex: Running to run) our words
#     text="".join(text) # Make our tokenize into sentences
    
    texts.append(i) #appending to list

from sklearn.feature_extraction.text import CountVectorizer # need to turn into vectors

count = CountVectorizer(stop_words = "english", max_features = 550) #it removes unnecesarry words and finds most using 550 words
matrix = count.fit_transform(depression_df.clean_text).toarray()

# Set X and Y
X = matrix
y = depression_df["is_depression"].values # values for turning series to array

# Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)






# In[]

print('\n_______________________________________\n')

#Predict With Naive-Bayes
from sklearn.naive_bayes import GaussianNB

# naive Bayse 

nb = GaussianNB()
nb.fit(X_train,y_train)


#GaussianNB() 
print ('The depression model accuracy - Naive Bayes: ', round(nb.score(X_test,y_test),3))


from sklearn.metrics import confusion_matrix
print ('\n Confusion Matrix:\n ', confusion_matrix(y_test,nb.predict(X_test)))


#try the model (predict)

# first we have to make preprocessing 
user_text_str = str(user_text).lower()
user_text= re.sub("[^a-zA-Z]"," ",user_text_str)                    # it removes unalfabetic signs
user_text = nltk.word_tokenize(user_text,language="english")        # it tokenizes our words
user_text=[lemmatize.lemmatize(word) for word in user_text_str]     # it lemmatizes our words
user_text = "".join(user_text) # Make our tokenize into sentences

# now let's do it and make prediction on text 
# t = 'I am happy'
texts.insert(0,user_text)
user_text = count.fit_transform(texts).toarray()
is_depressed_NB = nb.predict([user_text[0]])

print ('\n The result of text depression detection Using Naive Bayse:', is_depressed_NB ) 



# In[] 

print('\n_______________________________________\n')

# Support Vector Machine  

from sklearn.svm import SVC

# Train the SVM classifier
svm= SVC(kernel='linear') # poly -> 72 , sigmoid -> 83 # 
svm.fit(X_train, y_train)



print ('The depression model accuracy - SVC: ', round(svm.score(X_test,y_test),3))
print ('\n Confusion Matrix:\n ', confusion_matrix(y_test,svm.predict(X_test)))


# Predict the classes of the validation set
is_depressed_SVM = svm.predict([user_text[0]])

print ('\n The result of text depression detection using SVM:', is_depressed_SVM) 




# In[]

print('\n_______________________________________\n')

from sklearn.linear_model import LogisticRegression


# Train the logistic regression classifier



lr = LogisticRegression()
lr.fit(X_train, y_train)

print ('The depression model accuracy - Logistic Regression: ', round(lr.score(X_test,y_test),3))
print ('\n Confusion Matrix:\n ', confusion_matrix(y_test,lr.predict(X_test)))


# Predict the classes of the validation set
is_depressed_LR = lr.predict([user_text[0]])

print ('\n The result of text depression detection using Logistic Regression:', is_depressed_LR) 



# In[]

print('\n_______________________________________\n')

from sklearn.neural_network import MLPClassifier


# Train the Neural Network classifier
nn= MLPClassifier(activation ='relu', alpha=0.005,hidden_layer_sizes=(5,8),max_iter=1000) # identity -> 92.1  logistic -> 92.4    relu -> 92.6
nn.fit(X_train, y_train)

print ('The depression model accuracy - Neural Network: ', round(nn.score(X_test,y_test),3))
print ('\n Confusion Matrix:\n ', confusion_matrix(y_test,nn.predict(X_test)))


# Predict the classes of the validation set
is_depressed_NN = nn.predict([user_text[0]])

print ('\n The result of text depression detection using Neural Network:', is_depressed_NN) 






# In[]

# Decision Tree Classifier

print('\n_______________________________________\n')

from sklearn.tree import DecisionTreeClassifier

# create a decision tree classifier and fit it to the training data
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)


print ('The depression model accuracy - Decision Tree: ', round(dt_clf.score(X_test,y_test),3))
print ('\n Confusion Matrix:\n ', confusion_matrix(y_test , dt_clf.predict(X_test)))


# Predict the classes of the validation set
is_depressed_DT = dt_clf.predict([user_text[0]])

print ('\n The result of text depression detection using Decision Tree:', is_depressed_DT) 



# In[]

# Random Forest Classifier

print('\n_______________________________________\n')

from sklearn.ensemble import RandomForestClassifier


# create a random forest classifier and fit it to the training data
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

print ('The depression model accuracy - Random Forest: ', round(rf_clf.score(X_test,y_test),3))
print ('\n Confusion Matrix:\n ', confusion_matrix(y_test,rf_clf.predict(X_test)))


# Predict the classes of the validation set
is_depressed_RF = rf_clf.predict([user_text[0]])

print ('\n The result of text depression detection using Random Forest:', is_depressed_RF) 



# In[]

# KNN Classifier

print('\n_______________________________________\n')

from sklearn.neighbors import KNeighborsClassifier


# create a KNN classifier and fit it to the training data
k = 5 # number of nearest neighbors to consider
knn_clf = KNeighborsClassifier(n_neighbors=k)
knn_clf.fit(X_train, y_train)


print ('The depression model accuracy - KNN: ', round(knn_clf.score(X_test,y_test),3))
print ('\n Confusion Matrix:\n ', confusion_matrix(y_test,knn_clf.predict(X_test)))


# Predict the classes of the validation set
is_depressed_KNN = knn_clf.predict([user_text[0]])

print ('\n The result of text depression detection using KNN:', is_depressed_KNN) 


