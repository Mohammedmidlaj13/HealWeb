from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import re
import nltk
import tensorflow
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')


app = Flask(__name__)
y = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
lb = LabelEncoder()
Y = lb.fit_transform(y)

model = tensorflow.keras.models.load_model(r"C:\Users\midla\PycharmProjects\pythonProject\emotion.h5")

model.make_predict_function()
from tensorflow.keras.preprocessing.text import one_hot
acount = 0
jcount = 0
scount = 0
fcount = 0
lcount = 0
sucount = 0

def sentence_cleaning(sentence):
    """Pre-processing sentence for prediction"""
    stemmer = PorterStemmer()
    stopword = set(stopwords.words('english'))
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopword]
    text = " ".join(text)
    corpus.append(text)
    #
    one_hot_word = [one_hot(input_text=word, n=len(text)) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word,maxlen=300,padding='post')
    return pad
def predict_label( paragraph ):
    sentences = paragraph.split(".")
    global acount, jcount ,scount, fcount ,lcount, sucount
    j = 0
    sad = 0
    a = 0
    lo = 0
    sup = 0
    fea = 0

    for sentence in sentences:
        print(sentence)
        sentence = sentence_cleaning(sentence)
        print(sentence)
        result = np.argmax(model.predict(sentence), axis=-1)
        result = lb.inverse_transform(result)[0]

        if result == "joy":
            j = j + 1
        elif result == "sadness":
            sad = sad + 1
        elif result == "anger":

            a = a + 1
        elif result == "love":
            lo = lo + 1
        elif result == "surprise":
            sup = sup + 1
        elif result == "fear":
            fea = fea + 1

        proba = np.max(model.predict(sentence))

        if(result == "anger"):
            acount=acount+1
        elif (result == "joy"):
            jcount = jcount + 1
        elif (result == "sadness"):
            scount = scount + 1
        elif (result == "fear"):
            fcount = fcount + 1
        elif (result == "surprise"):
            sucount = sucount + 1
        elif (result == "love"):
            lcount = lcount + 1
        count = max(acount, scount, sucount, lcount, fcount, jcount)
        if(count>2):
            print("You have been experiencing more ",result," lately" )
            acount = 0
            jcount = 0
            scount = 0
            fcount = 0
            lcount = 0
            sucount = 0




        print(f"{result} : {proba}\n\n")
    emotions = {'joy': j, 'sadness': sad, 'anger': a, 'love': lo, 'surprise': sup, 'fear': fea}
    max_emotion = max(emotions, key=emotions.get)
    day_emotion = max_emotion
    print("The most frequent emotion is", max_emotion, "with", emotions[max_emotion], "occurrences.")

    return [max_emotion,count]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("heal.html")

@app.route("/about")
def about_page():
    return "Please Input Correctly"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        name = request.form.get('journal')
        p = predict_label(name)
        emo=p[0]
        c=p[1]


    return render_template("Result.html",emotion = emo,count=c)

@app.route("/book", methods = ['GET', 'POST'])
def booking():
    if request.method == 'POST':
        return render_template("index.html")


if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)