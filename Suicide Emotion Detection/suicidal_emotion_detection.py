
# import needed packajes 
import pandas as pd                     # to read csv files.
from stop_words import get_stop_words   # or we can use from nltk.corpus import stopwords. 
from nltk.tokenize import word_tokenize # it turn text to list but more faster. 
import re                               # to remove all non-alphabetic and punctuation signs  
from nltk.stem import WordNetLemmatizer # reducing words to their base or dictionary form,
from collections import Counter         # to count words in list and turn it to dictionary  
import matplotlib.pyplot as plt         # to plot emotions 
from wordcloud import WordCloud         # wordcloud ploting 



# function to preprocess text 
def preprocess_text(text):
    '''
    this function take text and pre-process it then turn it to list of words 

    Parameters
    ----------
    text : string 

    Returns
    -------
    text_list : list of text words after cleaning.

    '''
    # A. turn letters into lowercase 
    text = text.lower()
        

    # B. remove all non-alphabetic characters and punctuation marks
    text = re.sub(r'[^a-zA-Z\s]', '', text)
        
    # C. Tokenizaiton
    text = word_tokenize(text,"english")
    
    # D. Remove stop words but what are the stop words 
    stop_words = get_stop_words('english') 
    text_list = [word for word in text if word not in stop_words]


    # E. after tokenization and stop words removal it's tiem for lemmatization
    #lemmatizer = WordNetLemmatizer()
    #text_list = [lemmatizer.lemmatize(word) for word in text_list]

    # return the list of words
    return text_list

# function to read emotions from file 
def emotion_maping (directory): 
    
    emotion_dict = {}
    
    file = open(directory,'r',encoding='utf-8') 
    
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        emotion_dict[word] = emotion
    
    file.close()
    return emotion_dict

# function to read speeches from file and count emotions  
def get_emotions_from_speeches(directory,emotion_dict):
    
    # for emotions 
    temp_suicide_emotions_list = []

    # for suicide data 
    suicide_df = pd.read_csv(directory,index_col= 0)

    # read just suicidal speeches - 116037 records
    suicide_df = suicide_df[suicide_df['class'] == 'suicide']

    # reset the records indices
    suicide_df.reset_index(inplace = True, drop = True)

    for s in range(len(suicide_df)):

        text = str(suicide_df['text'][s])
        cleaned_text_list = preprocess_text(text)
        
        # suicide emotion list builder
        for word in emotion_dict.keys():
            if word in cleaned_text_list:
                temp_suicide_emotions_list.append(word)
    
    # count emotions 
    suicide_emotions_score_dict = dict(Counter(temp_suicide_emotions_list))
    return suicide_emotions_score_dict

# function to plot findings 
def plot_findings(speeches_emotions):
    
    #___________________________________ Plot for +1000 suicide emotions ___________________________________________________

    # turn the emotion list into dictionary 

    # now let's go more Specifically and just choose emotions with freq more than 10
    plus_1000_emotions = dict( (key, value) for (key, value) in speeches_emotions.items() if value >= 1000 )

    fig, ax = plt.subplots()
    ax.bar(plus_1000_emotions.keys(), plus_1000_emotions.values())

    # the fit the text undre graph 
    fig.autofmt_xdate()

    # change the width and length of plot 
    fig.set_figwidth(50)
    fig.set_figheight(25)

    # label the plot 
    plt.title('Frequent emotions in suicide speech', pad= 10 ,fontsize = 30)
    plt.xlabel('Emotions')
    #plt.ylabel('Frequency')

    # save the graph and show it 
    plt.savefig('suicide_1000+_emotions.png')
    plt.show()


    # --------------------------------------- plot suicide emotions using word cloud --------------------------------- 


    # generate for string , generate from frequences for dictionaries
    wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=110).generate_from_frequencies(suicide_emotions_score_dict) 

    plt.figure(figsize=(10, 7)) 
    plt.imshow(wordcloud, interpolation="bilinear") 
    plt.axis('off')

    # save the graph and show it 
    plt.savefig('suicide_1000+_emotions_wordcloud.png') 
    plt.show()
            

# function to search on any text for red flags
def is_red_flag_text (red_flags_emotions):
    # test your text 
    text = input("Enter Ur text to test red-flags: \n")

    # check for red flags 
    red_flag = False

    for word in text.split():
        if word in red_flags_emotions:
            red_flag = True 
            break
        
    if(red_flag):
        print ("There are red flag emotion in your text !")
    else:
        print ("No Worry , Your text is fine ")


# ------------------------------------------------------------ Main Section - Read Emotions and speeches files  ----------------------------------------------------------


# get the emotion dictionary ready
emotions_file ='depression_emotions_original.txt'
emotion_dict = emotion_maping(emotions_file)
suicide_emotions_score_dict = get_emotions_from_speeches('Suicide_Detection.csv',emotion_dict)
plot_findings(suicide_emotions_score_dict)
red_flags_emotions = ['lost','scared','afraid','hurt','alone','lonely','suffering','depressed','rejected','terrified']


# test random text 
is_red_flag_text(red_flags_emotions)

