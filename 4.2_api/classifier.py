import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import swifter
import sys
import nltk
from sklearn.linear_model import LogisticRegression
import pandas as pd
from unidecode import unidecode
import pickle
from flask import Flask, request, Response
import jsonpickle

app = Flask(__name__)
app.config["DEBUG"] = True


stemmer = SnowballStemmer("english")
stop = set(stopwords.words('english'))
def lower(texto):
    return texto.lower()

def normalize(texto):
    return unidecode(texto)

def remove_ponctuation(texto):
    for punc in string.punctuation:
        texto = texto.replace(punc," ")
    return texto

def remove_stopwords(texto):
    ret = []
    for palavra in texto.split():
        if palavra not in stop:
            ret.append(palavra)
    return ' '.join(ret)

def stem(texto):

    ret = []
    for palavra in texto.split():
        ret.append(stemmer.stem(palavra))
    return ' '.join(ret)


def remove_number(texto):
    result = ''.join([i for i in texto if not i.isdigit()])
    return result

def pipeline(texto):
    texto = normalize(texto)
    texto = lower(texto)
    texto = remove_ponctuation(texto)
    texto = remove_stopwords(texto)
    texto = remove_number(texto)
    texto = stem(texto)
    return texto


def classify_text(text,clf,vectorizer):
	text = pipeline(str(text))
	x = [text]
	x = vectorizer.transform(x)
	return clf.predict(x)


infile = open('clf.pickle','rb')
clf = pickle.load(infile)
infile.close()

infile = open('vectorizer.pickle','rb')
vectorizer = pickle.load(infile)
infile.close()


@app.route('/sentimento/<texto>', methods=['GET'])
def api_class(texto):
	print(texto)
	resp = classify_text(texto,clf,vectorizer)
	response={}
	if(resp[0] == 1):
		response.update({'Sentiment':'positive'})
	else:
		response.update({'Sentiment':'negative'})
	response_pickled = jsonpickle.encode(response)
	return Response(response=response_pickled, status=200, mimetype="application/json")





if __name__ == '__main__':
	app.run(host="localhost", port=8765)