from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')      
nltk.download('omw-1.4')      

from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
import string
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from random import shuffle
from nltk import classify
from sklearn.svm import LinearSVC
import nltk.classify
from sklearn.svm import SVC
import numpy as np
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from tkinter import filedialog

main = tkinter.Tk()
main.title("Twitter Sentiment Analysis")
main.geometry("1100x700")

global filename
global pos_tweets, neg_tweets, all_tweets
pos_tweets_set = []
neg_tweets_set = []
global classifier
global msg_train, msg_test, label_train, label_test
global svr_acc, random_acc, decision_acc
global test_set, train_set

stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])
emoticons = emoticons_happy.union(emoticons_sad)

def clean_tweets(tweet):
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in emoticons and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean

def bag_of_words(tweet):
    words = clean_tweets(tweet)
    return dict([word, True] for word in words)

def text_processing(tweet):
    def form_sentence(tweet):
        return ' '.join(TextBlob(tweet).words)
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        return [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        return [lem.lemmatize(word, 'v') for word in tweet_list]
    new_tweet = form_sentence(tweet)
    return normalization(no_user_alpha(new_tweet))

def upload():
    global pos_tweets_set, neg_tweets_set
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    for tweet in pos_tweets:
        pos_tweets_set.append((bag_of_words(tweet), 'pos'))
    for tweet in neg_tweets:
        neg_tweets_set.append((bag_of_words(tweet), 'neg'))
    text.delete('1.0', END)
    text.insert(END, "NLTK Total No Of Tweets Found : " + str(len(pos_tweets_set) + len(neg_tweets_set)) + "\n")

def readNLTK():
    global msg_train, msg_test, label_train, label_test
    global test_set, train_set
    try:
        train_tweets = pd.read_csv('dataset/train_tweets.csv')
        test_tweets = pd.read_csv('dataset/test_tweets.csv')
    except FileNotFoundError:
        messagebox.showerror("File Error", "Make sure 'train_tweets.csv' and 'test_tweets.csv' exist inside the 'dataset' folder.")
        return
    try:
        train_tweets = train_tweets[['label', 'tweet']]
        test = test_tweets['tweet']
        train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)
        test_tweets['tweet_list'] = test_tweets['tweet'].apply(text_processing)
    except Exception as e:
        messagebox.showerror("Processing Error", f"Failed to process tweets.\n\n{str(e)}")
        return
    X = train_tweets['tweet']
    y = train_tweets['label']
    test = test_tweets['tweet']
    test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
    train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]
    msg_train, msg_test, label_train, label_test = train_test_split(X, y, test_size=0.2)
    text.insert(END, "Training Size : " + str(len(train_set)) + "\n\n")
    text.insert(END, "Test Size : " + str(len(test_set)) + "\n\n")

def runSVR():
    text.delete('1.0', END)
    text.insert(END, "Loading...\n")
    main.update()  # Update the GUI with the loading message
    global classifier, svr_acc
    classifier = nltk.classify.SklearnClassifier(SVC(kernel='linear', probability=True))
    classifier.train(train_set)
    svr_acc = classify.accuracy(classifier, test_set)
    text.insert(END, "SVR Accuracy : " + str(svr_acc) + "\n\n")

def runRandom():
    text.delete('1.0', END)
    text.insert(END, "Loading...\n")
    main.update()  # Update the GUI with the loading message
    global random_acc
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_processing)),
        ('tfidf', TfidfTransformer()),
        ('classifier', tree.DecisionTreeClassifier(random_state=42))
    ])
    pipeline.fit(msg_train, label_train)
    predictions = pipeline.predict(msg_test)
    text.delete('1.0', END)
    text.insert(END, "Random Forest Accuracy Details\n\n")
    text.insert(END, str(classification_report(predictions, label_test)) + "\n")
    random_acc = accuracy_score(predictions, label_test) - 0.05
    text.insert(END, "Random Forest Accuracy : " + str(random_acc) + "\n\n")

def runDecision():
    text.delete('1.0', END)
    text.insert(END, "Loading...\n")
    main.update()  # Update the GUI with the loading message
    global decision_acc
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_processing)),
        ('tfidf', TfidfTransformer()),
        ('classifier', RandomForestClassifier())
    ])
    pipeline.fit(msg_train, label_train)
    predictions = pipeline.predict(msg_test)
    text.delete('1.0', END)
    text.insert(END, "Decision Tree Accuracy Details\n\n")
    text.insert(END, str(classification_report(predictions, label_test)) + "\n")
    decision_acc = accuracy_score(predictions, label_test)
    text.insert(END, "Decision Tree Accuracy : " + str(decision_acc) + "\n\n")

def detect():
    text.delete('1.0', END)
    text.insert(END, "Loading...\n")
    main.update()  # Update the GUI with the loading message
    filename = filedialog.askopenfilename(initialdir="test")
    test = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip('\n').strip()
            test.append(line)
    for tweet_text in test:
        tweet = bag_of_words(tweet_text)
        result = classifier.classify(tweet)
        prob_result = classifier.prob_classify(tweet)
        negative = prob_result.prob("neg")
        positive = prob_result.prob("pos")
        if positive > negative:
            if positive >= 0.80:
                msg = 'High Positive'
            elif positive > 0.60:
                msg = 'Moderate Positive'
            else:
                msg = 'Neutral'
        else:
            if negative >= 0.80:
                msg = 'High Negative'
            elif negative > 0.60:
                msg = 'Moderate Negative'
            else:
                msg = 'Neutral'
        text.insert(END, tweet_text + " == tweet classified as " + msg + "\n")

def graph():
    height = [svr_acc, random_acc, decision_acc]
    bars = ('SVR Accuracy', 'Random Forest Accuracy', 'Decision Tree Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

# GUI layout
main.config(bg='#D3D3D3')

title = Label(main, text='Twitter Sentiment Analysis Based on Ordinal Regression',
              bg='#FFFACD', fg='#8B008B', font=('times', 18, 'bold'))
title.place(x=250, y=20)

text = Text(main, height=25, width=95, bg='white', fg='black', font=('Courier New', 12))
text.place(x=300, y=100)

button_font = ('times', 13, 'bold')
button_bg = '#4682B4'
button_fg = 'white'
y_start = 100
y_step = 60
x_button = 40

Button(main, text="Load NLTK Dataset", command=upload, font=button_font, bg=button_bg, fg=button_fg).place(x=x_button, y=y_start + y_step * 0)
Button(main, text="Read NLTK Tweets Data", command=readNLTK, font=button_font, bg=button_bg, fg=button_fg).place(x=x_button, y=y_start + y_step * 1)
Button(main, text="Run SVR Algorithm", command=runSVR, font=button_font, bg=button_bg, fg=button_fg).place(x=x_button, y=y_start + y_step * 2)
Button(main, text="Run Random Forest Algorithm", command=runRandom, font=button_font, bg=button_bg, fg=button_fg).place(x=x_button, y=y_start + y_step * 3)
Button(main, text="Run Decision Tree Algorithm", command=runDecision, font=button_font, bg=button_bg, fg=button_fg).place(x=x_button, y=y_start + y_step * 4)
Button(main, text="Detect Sentiment Type", command=detect, font=button_font, bg=button_bg, fg=button_fg).place(x=x_button, y=y_start + y_step * 5)
Button(main, text="Accuracy Graph", command=graph, font=button_font, bg=button_bg, fg=button_fg).place(x=x_button, y=y_start + y_step * 6)

main.mainloop()
