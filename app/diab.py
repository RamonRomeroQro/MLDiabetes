from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import math
from numpy import genfromtxt
from sklearn.covariance import EllipticEnvelope
from sklearn import svm


app = Flask(__name__)

@app.route('/')
def hello_world():
   
    dictionary = {"PESO": "Peso en KG" , "ESTATURA": "Estatura en metros",
                "EDAD": "¿Qué edad tiene?","GENERO": "¿Cuál es su genero?",
                "PADRES": "¿Alguno de sus padres es diabético?",
                "HERMANOS": "Si tiene hemanos, ¿Alguno de sus hermanos es diabético?",
                "HIJOS":  "Si tiene hijos, ¿Alguno de sus padres es diabético?",
                "OTROS": "Tiene algun otro familiar diabético",
                "ACTIVIDAD_FISICA": "¿Realiza 30 min o mas de actividad fisica?",
                "TABAQUISMO": "¿Es fumador o lo ha sido?",
                "TUBERCULOSIS":"¿Tiene tuberculosis?",
                "CANCER": "¿Tiene cancer?",
                "OBESIDAD": "¿Tiene Obesidad?",
                "HIPERTENSION": "¿Es hipertenso?",
                "VIH": "¿Tiene VIH/SIDA?",
                "DEPRESION": "¿Ha sufrido de depresión (Diagnosticada?",
                "DISLIPIDEMIA": "¿Es dislipidémico?",
                "CARDIO": "¿Sufre de algun desorden cardio vascular?",
                "HEPATITIS": "¿Ha sufrido de hepatitis?",
                "NUTRIOLOGO": "¿Ha visitado al nutriologo en el ultimo año?",
                "OFTALMOLOGO": "¿Ha visitado al oftalmologo en el ultimo año?",
                "PODOLOGO": "¿Ha visitado al podologo en el ultimo año?",
                "MOTORA": "¿Sufre de alguna discapacidad motriz?",
                "VISUAL": "¿Sufre de alguna discapacidad visual?",
    }
    return render_template('index.html', dict=dictionary)

from os import listdir
from os.path import isfile, join
import pickle


from sklearn.externals import joblib

from sklearn.feature_extraction import DictVectorizer

from sklearn.externals.joblib import dump, load


def ml(arr):
    models = "./outputs"
    onlyfiles = [f for f in listdir(models) if isfile(join(models, f))]
    onlyfiles.remove("scaler.save")

    answers={}

    scaler_filename = "./outputs/scaler.save"
    scaler = joblib.load(scaler_filename) 

    
    

    for i in onlyfiles:
        res=""
        p="./outputs/"+i


        
        loaded_model  = load(p) 

        if i.startswith("select"):
            select = np.array(arr[:3])
            select = select.reshape(1,-1)
            select = scaler.transform(select)

            res= str(loaded_model.predict(select))
            try:
                res=res+' % '+str(loaded_model.predict_proba(select)) # + " " + str(loaded_model.coef_)
            except:
                pass
        else:
            full = np.array(arr)
            full = full.reshape(1,-1)
            res= str(loaded_model.predict(full))
            try:
                res=res+' % '+str(loaded_model.predict_proba(full))
            except:
                pass
        
        

        answers[i]=res



   
    return  answers


@app.route('/evaluate', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        # load model
        order= ["EDAD","GENERO","PADRES","HERMANOS","HIJOS",
        "OTROS","ACTIVIDAD_FISICA","TABAQUISMO","TUBERCULOSIS","CANCER",
        "OBESIDAD","HIPERTENSION","VIH","DEPRESION","DISLIPIDEMIA",
        "CARDIO","HEPATITIS","NUTRIOLOGO","OFTALMOLOGO","PODOLOGO",
        "MOTORA","VISUAL", "PESO","ESTATURA"]

        predict= [ float(request.form[x]) for x in order]

        a = ml(predict)
        lr=a["select_Logistic Regression.save"]

        return result(a, lr)


      


    else:
        return hello_world()



@app.route('/result')
def result(eval, lr):
    # [1] % [[0.20667865 0.79332135]] 
    lr = lr.split("%")
    lr = lr[1].strip().strip('[').strip(']').split(" ")
    prob={}

    prob["NO"]=lr[0]
    prob["SI"]=lr[1]

    return render_template('more.html', result=eval, ans=prob )

