import re, unicodedata
import pickle

import tensorflow as tf 

from flask import Flask
from flask import render_template
from flask import request


app = Flask(__name__)
white_model = tf.keras.models.load_model('white_model.h5')
#red_model = tf.keras.models.load_model('red_model.h5')

@app.route('/', methods=["GET"])
def home():
    return render_template("home.html")

@app.route('/red', methods=["GET"])
def red():
    return render_template("red.html")

@app.route('/white', methods=["GET"])
def white():
    return render_template("white.html")

@app.route('/process_white', methods=["POST"])
def process_white():

    with open('input_tokenizer_white.pkl', 'rb') as f:
        input_tokenizer_white = pickle.load(f)
    with open('input_tensor_white.pkl', 'rb') as f:
        input_tensor_white = pickle.load(f)
    with open('white_wine_df.pkl', 'rb') as f:
        white_wine_df = pickle.load(f)
    if (request.form.get("preferences") == '1'):
        sentence = request.form.get("sentence")
    else:
        sentence = request.form.get("item1") + ' ' +  request.form.get("item2")  + ' ' +  request.form.get("item3")  + ' ' +  request.form.get("item4")
    df_result, pred = predict(sentence, 
               input_tokenizer_white, input_tensor_white, white_wine_df, False)
    df_result = df_result.drop('variety', axis = 1)
    return render_template("process.html", text = pred, df =df_result.to_html(index=False))

""" @app.route('/process_red', methods=["POST"])
def process_red():

    with open('input_tokenizer_red.pkl', 'rb') as f:
        input_tokenizer_red = pickle.load(f)
    with open('input_tensor_red.pkl', 'rb') as f:
        input_tensor_red = pickle.load(f)
    with open('red_wine_df.pkl', 'rb') as f:
        red_wine_df = pickle.load(f)
    
    if (request.form.get("preferences") == '1'):
        sentence = request.form.get("sentence")
    else:
        sentence = request.form.get("item1") + ' ' +  request.form.get("item2")  + ' ' +  request.form.get("item3")  + ' ' +  request.form.get("item4")    
    df_result, pred = predict(sentence, 
               input_tokenizer_red, input_tensor_red, red_wine_df, True)
    df_result = df_result.drop('variety', axis = 1)
    return render_template("process.html", text = pred, df =df_result.to_html(index=False)) """

def unicode_to_ascii(s):
    return ''.join([c for c in unicodedata.normalize('NFD', s)
                        if unicodedata.category(c) != 'Mn'])

def recommendation(df, name):
    return df[df['variety'] == name].sort_values(by='points', ascending=False)[:5]

def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
    
    s = re.sub(r"([?.!;,:()\"])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)    
    s = re.sub(r"[^a-zA-Z?.!;,:()\"]+", " ", s)    
    s = s.rstrip().strip()    
    s = '<start> ' + s + ' <end>'
    
    return s

def predict(sentence, input_tokenizer, input_tensor, df, is_red=True):
    sentence_tokens = preprocess_sentence(sentence)
    sentence_tokens = input_tokenizer.texts_to_sequences([sentence_tokens])
    sentence_length = input_tensor.shape[1]

    for i, s in enumerate(sentence_tokens):
        sentence_tokens[i] = s + ([0] * (sentence_length - len(s)))        
        
    white_classes = ['Chardonnay',  'Riesling', 'Sauvignon Blanc']
    red_classes = ['Cabernet Sauvignon', 'Pinot Noir', 'Red Blend Comb']
    if (is_red):        
        result = white_model.predict(sentence_tokens)
        result_text = "For your input '{0}', your recommended variety is: {1} with {2} confidence".format(sentence, red_classes[result[0].argmax()], result[0].max())    
        re_df = recommendation(df, red_classes[result[0].argmax()])
    else:
        result = white_model.predict(sentence_tokens)
        result_text = "For your input '{0}', your recommended variety is: {1} with {2} confidence".format(sentence, white_classes[result[0].argmax()], result[0].max())
        re_df = recommendation(df, white_classes[result[0].argmax()])
    return re_df, result_text


if __name__ == "__main__":
    app.run(debug=True)