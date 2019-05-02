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

def ml(predict):
    '''Probabilistic neural network'''
    # read data
    a = pd.read_csv('./data/2016.csv',header=0)
    b = pd.read_csv('./data/2016_2.csv', header=0)
    c = pd.read_csv('./data/2017.csv', header=0)
    d = pd.read_csv('./data/2018.csv', header=0)
    # key atrs
    important = ["IDE_EDA_ANO","IDE_SEX","DIAB_PAD_MAD","DIAB_HER","DIAB_HIJ",
    "DIAB_OTROS","CVE_ACT_FIS","CVE_TAB","CVE_COMB_TUBER",
    "CVE_COMB_CANCER","CVE_COMB_OBESIDAD","CVE_COMB_HIPER",
    "CVE_COMB_VIH_SIDA","CVE_COMB_DEPRE","CVE_COMB_DISLI","CVE_COMB_CARDIO",
    "CVE_COMB_HEPA","CVE_NUT","CVE_OFT",
    "CVE_PIES","CVE_DIAB","CVE_TIPO_DISC_MOTO","CVE_TIPO_DISC_VISU",
    "PESO","ESTATURA"]




    # label

    label="CVE_DIAB"
    # vertical join

    result = pd.concat([a, b, c, d])

    # hot encoding

    result= result.replace("Masculino", 1)
    result= result.replace("Femenino", 0)
    result= result.replace("Si", 1)
    result= result.replace("No", 0)
    result['CVE_DIAB'] = result['CVE_DIAB'].replace(0,2)
    result["CVE_TAB"]  = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_TAB"])  else 1,
                        axis=1)

    result.fillna(0)
    result["CVE_NUT"]  = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_NUT"])  else 1,
                        axis=1)

    result["CVE_OFT"]  = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_OFT"])  else 1,
                        axis=1)

    result["CVE_PIES"]  = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_PIES"])  else 1,
                        axis=1)
    result= result[important]


    # norm parameters
    min_norm_edad =  result['IDE_EDA_ANO'].min()
    max_norm_edad =  result['IDE_EDA_ANO'].max()
    min_norm_estatura = result['ESTATURA'].min()
    max_norm_estatura = result['ESTATURA'].max()
    min_norm_peso = result['PESO'].min()
    max_norm_peso = result['PESO'].max()



    select = result.loc[result.CVE_DIAB == 1]
    print(select)
    fx= ["PESO","ESTATURA"]
    #X = np.array([ result[x].tolist() for x in fx  ].append(]) ).T

    a = [ select[x].tolist() for x in fx  ]
    a[0].append(predict[-2])
    a[1].append(predict[-1])
    X = np.array(a).T
    outliers_fraction = 0.15
    sout= svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)
    sout.fit(X)
    ok1 = sout.predict(X)
    sv = ok1[-1]



    outlier = EllipticEnvelope(contamination=0.1)

    outlier.fit(X)
    prediction1 = outlier.predict(X)
    eliptic = prediction1[-1]
                    
    # Normalization         
    cols_to_norm = [ "PESO"  , "ESTATURA", "IDE_EDA_ANO"]
    result[cols_to_norm] = result[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    #print(result.to_string())
    important.remove(label)
    cleaned_data = np.array([ result[x].tolist() for x in important ]).T

    




    # print (result)
    # print (cleaned_data)
    # print(result.to_string())

    groups = result.groupby(label)
    number_of_classes = len(groups)  # Here we have 3 different classes
    dictionary_of_sum = {}
    numrber_of_features  = len(result.columns) -1 # We have feature 1 and feature 2 
    sigma = 1
    increament_current_row_in_matrix = 0



    point_want_to_classify = predict
    
    point_want_to_classify[0] = (point_want_to_classify[0]  - float(min_norm_edad)) / (float(max_norm_edad) - float(min_norm_edad))
    point_want_to_classify[-1] = (point_want_to_classify[-1]  - float(min_norm_estatura)) / (float(max_norm_estatura) - float(min_norm_estatura))
    point_want_to_classify[-2] = (point_want_to_classify[-2]  - float(min_norm_peso)) / (float(max_norm_peso) - float(min_norm_peso))
    print("xxx",point_want_to_classify[0], float(result['IDE_EDA_ANO'].min()))
    print("!!!!", point_want_to_classify)
    
    for k in range(1,number_of_classes+1):

        # 4.1 Initiate the sume to zero 
        dictionary_of_sum[k] = 0
        number_of_data_point_from_class_k = len(groups.get_group(k))

     
        temp_summnation = 0.0

        for i in range(1,number_of_data_point_from_class_k+1):

            # Gaussian sum
            temparr=[]
            for index in range(len(point_want_to_classify)-1):

                tempx = math.pow((point_want_to_classify[index] - cleaned_data[increament_current_row_in_matrix][index]),2)
                #print("->",tempx)
                #print("n",increament_current_row_in_matrix)
                temparr.append(tempx)
            
            temp_sum = -1 * (sum(temparr))
            #print("!!",temp_sum) 
            temp_sum = temp_sum/( 2 * np.power(sigma,2)  )
            #print("!!-",temp_sum) 

            # Sum of Gaussians

            
            temp_summnation = temp_summnation + math.e**temp_sum 
            #print("!!+",temp_summnation) 

            increament_current_row_in_matrix  = increament_current_row_in_matrix + 1

        dictionary_of_sum[k]  = temp_summnation 

    classified_class = str( max(dictionary_of_sum, key=dictionary_of_sum.get) )
    return   (dictionary_of_sum, classified_class, eliptic, sv)


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
        print(predict)
        ev , c, e, sv= ml(predict)
        print(ev, c, e, sv)
        return result(sv)


      


    else:
        return hello_world()



@app.route('/result')
def result(eval):
    return render_template('more.html', result=eval)

