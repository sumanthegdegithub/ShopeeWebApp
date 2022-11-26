from flask import Flask, render_template, request, jsonify, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
import string
import tensorflow as tf
from tensorflow import image
import nltk
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
nltk.download('wordnet')
nltk.download('omw-1.4')
import joblib
import tensorflow_text as text


interpreter = tf.lite.Interpreter(model_path="image_model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tx = tf.keras.models.load_model('text_model')
images_search_model = joblib.load('image_search.sav')
text_search_model = joblib.load('text_search.sav')


UPLOAD_FOLDER = '/static/imgs'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dataset = pd.read_csv('test_dataset.csv')

def product_card(imagename, text, pid, page):
    
    a = open("templates/product_card.html", "r")
    a = str(a.read())
    if page == 'product':
        a = a.replace('product_link_placeholder', str(pid))
    elif page == 'index':
        a = a.replace('product_link_placeholder', 'product/'+str(pid))
    a = a.replace('image_name_placeholder', 'https://shopeeimages.blob.core.windows.net/newcontainer/'+imagename)
    a = a.replace('text_placeholder', text)
    
    return a

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
image_size = [224,224] #We will convert all the images to 224x224x3 size

def decode_image(image_data):
  '''
  This function takes the location of a image data
  converts it and returns tensors
  '''
  img = tf.io.read_file(image_data)
  img = tf.io.decode_jpeg(img, channels = 3)
  img = image.resize(img, image_size)
  img = tf.cast(img, tf.float32) / 255.0
  img = tf.reshape(img, [-1, 224, 224, 3])
  return img

def preprocessing(text):
  '''
  This function takes raw text Converts uppercase to Lower, Removes special charecters, stop words 
  and text that follows special charecters (Ex \xe3\x80\x80\)
  '''

  text=text.strip() #Strips the sentence into words
  text = text.lower() #Converts uppercase letters to lowercase
  text = text.split(' ') #Splitting sentence to words
  dummy = []
  for i in text:
    if i[:2] == '\\x':
      special_texts = [i.split("\\") for i in text if i[:2] == '\\x']
      for j in special_texts[0]:
        if len(j) > 3:
          dummy.append(j[3:])
    else:
      dummy.append(i)
  text = dummy
  stopwords = get_stop_words('english') + get_stop_words('indonesian') #Getting stopwords from english and indonesian languages
  text = [i for i in text if i not in stopwords] ##Removing stopwords
  wordnet_lemmatizer = WordNetLemmatizer() #Loading Lemmetizer
  text = [wordnet_lemmatizer.lemmatize(word) for word in text] #Lemmetizing the text
  text = " ".join([i for i in text if len(i) < 15]) #Remove the words longer than 15 charecters
  text = "".join([i for i in text if i not in string.punctuation]) #Removing special charecters
  return text

@app.route('/')
def index():
    f = open("templates/index.html", "r")
    f = str(f.read())
    
    return f

@app.route('/product/<string:pid>', methods=['GET'])
def product_page(pid):
    
    imagename = str(list(dataset.image[dataset.posting_id == pid])[0])
    text = str(list(dataset.title[dataset.posting_id == pid])[0])
    f = open("templates/product.html", "r")
    f = str(f.read())
    f = f.replace('image_name_placeholder', 'https://shopeeimages.blob.core.windows.net/newcontainer/'+imagename)
    f = f.replace('text_placeholder', text)
    
    similarProducts = dataset.similar_products.values[dataset.posting_id == pid][0]
    similarProducts = similarProducts.strip("][").split(', ')
    similarProducts = [i.strip("'") for i in similarProducts]
    random_products = dataset[dataset.posting_id.isin(similarProducts)]
    
    div = ''
    for i in range(len(random_products)):
        imagename = list(random_products.image)[i]
        text = list(random_products.title)[i]
        pid = list(random_products.posting_id)[i]
        if len(text) > 50:
            text = text[:50]+'...'
        else:
            text = text + '...' + ' '*(50-len(text))
        div+=product_card(imagename, text, pid, 'product')
    
        
    f = f.replace('products_place_holder', div)
    return f

@app.route('/search', methods=['POST'])
def search_page():
    if request.method == 'POST':
        indices = []
        
        text = request.form['text']
        print(text)
        if len(text) > 0:
            text = preprocessing(text)
            
            t = tx.predict([text])
            text_distances, text_indices = text_search_model.kneighbors(t)
            indices = text_indices[0][:10].tolist()
        
        f = open("templates/searchresults.html", "r")
        f = str(f.read())
        if 'file' not in request.files:
            if len(indices) > 0:
                pass
            
        file = request.files['file']
        if file.filename == '':
            if len(indices) > 0:
                pass
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save('static/imgs/'+filename)
            
            img = decode_image('static/imgs/'+filename)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            l = interpreter.get_tensor(output_details[0]['index'])
            image_distances, image_indices = images_search_model.kneighbors(l)
            
            if len(indices)>0:
                indices = indices[:5]
                indices.extend(image_indices[0][:5].tolist())
            else:
                indices.extend(image_indices[0][:10].tolist())
            
        
        imagename_list = list(dataset.image)
        text_list = list(dataset.title)
        pid_list = list(dataset.posting_id)
        div = ''
        if len(indices) == 0:
            f = f.replace('products_place_holder', '<center>Enter atleast a query text or an image to search<center>')
        else:
            for i in indices:
                imagename = imagename_list[i]
                text = text_list[i]
                pid = pid_list[i]
                if len(text) > 50:
                    text = text[:50]+'...'
                else:
                    text = text + '...' + ' '*(50-len(text))
                div+=product_card(imagename, text, pid, 'index')
                
            f = f.replace('products_place_holder', div)

        return f

if __name__ == '__main__':
    app.debug = False
    app.run()
    

