import pickle
from flask import Flask, request, render_template

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

## ! import ridge regressor and standard scaler pickle.
ridge_model=pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler=pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
      return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_performance():
      if request.method=='POST':
            HS=int(request.form.get('Hours Studied'))
            PS=int(request.form.get('Previous Scores'))
            EA=int(request.form.get('Extracurricular Activities'))
            SH=int(request.form.get('Sleep Hours'))
            SQ=int(request.form.get('Sample Question Papers Practiced'))

            new_data_scaled=standard_scaler.transform([[HS, PS, EA, SH, SQ]])

            result=ridge_model.predict(new_data_scaled)

            return render_template('home.html', results=result[0])
      else:
            return render_template('home.html')

if __name__ == '__main__':
      app.run(host='0.0.0.0')