from flask import Flask,url_for,request,render_template
import os
from keras.models import load_model
import cv2
import numpy as np
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
@app.route('/')
def intro():
    return render_template('intro.html')
@app.route('/result',methods = ['POST'])
def result():
    target = os.path.join(APP_ROOT,'img/')
    if not os.path.isdir(target):
        os.mkdir(target)
    
    for img in request.files.getlist('input') :
        
        filename = img.filename
        img.save(str(target)+'/'+str(filename))
        img = cv2.imread(str(target)+'/'+str(filename))
        img = cv2.resize(img,(32,32))
        img = np.expand_dims(img,axis=0)
        model = load_model('weights.h5')
        pred = model.predict(img)
        pred = int(np.argmax(pred))
            
    return render_template('output.html',output = pred)


app.run(debug=True)