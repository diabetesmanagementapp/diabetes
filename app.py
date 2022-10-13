
from statistics import mode
from unittest import result
from flask import Flask,request
from flask import jsonify 
import pickle
import numpy as np

model=pickle.load(open('diabetes_model.pkl','rb'))

app=Flask(__name__)
@app.route('/')
def welcome():
    return " world"

@app.route('/predict',methods=['POST'])
def predict():
    preg=request.form.get('preg')
    bp=request.form.get('bp')
    bmi=request.form.get('bmi')
    age=request.form.get('age')
  

   
    input_data=np.array([[preg,bp,bmi,age]])
    result=model.predict(input_data)[0]

    return jsonify({'result':str(result)})

if __name__ == '__main__':
 app.run(host="0.0.0.0",port=5000)






