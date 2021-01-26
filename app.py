import numpy as np
from flask import Flask, render_template, request, redirect
import pickle

#intialize the flask app
app = Flask(__name__, template_folder="templates")
model = pickle.load(open("model.pkl","rb"))

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

#Now predict function
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on the HTML page
    prg = request.form['prg']
    glc = request.form['gl']
    bp = request.form['bp']
    skt = request.form['sk']
    ins = request.form['ins']
    bmi = request.form['BMI']
    dpf = request.form['ped']
    age = request.form['age']

    prg = int(prg)
    glc = int(glc)
    bp = int(bp)
    skt = int(skt)
    ins = int(ins)
    bmi = float(bmi)
    dpf = float(dpf)
    age = int(age)

    final_features = np.array([(prg, glc, bp, skt, ins, bmi,dpf, age)])
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text = "The patient has diabetes : {}".format(prediction))

if __name__=="__main__":
    app.run(debug = True)
