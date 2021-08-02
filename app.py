#Important Modules
from flask import Flask,render_template, url_for ,flash , redirect
#from forms import RegistrationForm, LoginForm
import joblib
from flask import request
import numpy as np
from sklearn import *
import tensorflow
#from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
#from flask_sqlalchemy import SQLAlchemy
#from model_class import DiabetesCheck, CancerCheck

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
#from tensorflow.keras.layers import GlobalMaxPooling2D, Activation
#from tensorflow.keras.layers.normalization import BatchNormalization
#from tensorflow.keras.layers.merge import Concatenate
#from tensorflow.keras.models import Model

import pandas as pd

import pickle

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf


app=Flask(__name__,template_folder='template')

model_cancer = pickle.load(open('model_cancer.pkl', 'rb'))


model_kidney = pickle.load(open('model_kidney.pkl', 'rb'))

model_liver = pickle.load(open('model_liver.pkl', 'rb'))


model_dia = pickle.load(open('model_dia.pkl', 'rb'))
dataset_diab = pd.read_csv('diabetes.csv')
dataset_X = dataset_diab.iloc[:,[0,1,2,3,4,5,6,7]].values
from sklearn.preprocessing import MinMaxScaler
sc_diab = MinMaxScaler(feature_range = (0,1))
dataset_scaled_diab = sc_diab.fit_transform(dataset_X)





dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

#graph = tf.get_default_graph()
#with graph.as_default():;
from tensorflow.keras.models import load_model
model = load_model('model111.h5')

#FOR THE FIRST MODEL

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted





# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))




@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")
 


@app.route("/about")
def about():
    return render_template("about.html")



@app.route("/kidneypred")
def kidneypred():
    #if form.validate_on_submit():
    return render_template("kidneypred.html")

@app.route("/Malaria")
def Malaria():
    return render_template("index.html")

@app.route("/dia")
def dia():
    return render_template("indexdia.html")

@app.route("/heartpred")
def heartpred():
    return render_template("original.html")

@app.route("/liverpred")
def liverpred():
    return render_template("kiverpredict.html")

@app.route("/cancerpred")
def cancerpred():
    return render_template("cancer_pred.html")

@app.route('/predictdia',methods=['POST'])
def predictdia():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_dia.predict( sc_diab.transform(final_features) )

    if prediction == 1:
        pred = "It seems like you have Diabetes. Please consult a Doctor."
    elif prediction == 0:
        pred = "It seems you are fine. You don't have Diabetes."
    output = pred

    return render_template('indexdia.html', prediction_text='{}'.format(output))


@app.route('/predict2',methods=['POST'])
def predict2():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs= float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

        mul_reg = open('model_heart.pkl','rb')
        ml_model = joblib.load(mul_reg)
        model_predcition = ml_model.predict([pred_args])
        if model_predcition == 1:
            res = 'It seems like you have a Heart Disease. Please consult a Doctor.'
        else:
            res = "It seems you are fine. You don't have a Heart Disease."
        #return res
    return render_template('original.html', prediction = res)

@app.route("/predict3", methods=['POST'])
def predict3():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = float(request.form['Gender'])
        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])


        values = np.array([[Age,Gender,Total_Bilirubin, Direct_Bilirubin ,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
        prediction = model_liver.predict(values)

        return render_template('kiverpredict.html', prediction=prediction)


@app.route("/predict4", methods=['POST'])
def predict4():
    if request.method == 'POST':
        Age = float(request.form['age'])
        BP = float(request.form['bp'])
        AL=float(request.form['al'])
        SU= float(request.form['su'])
        RBC = float(request.form['rbc'])
        PC = float(request.form['pc'])
        PCC = float(request.form['pcc'])
        BA = float(request.form['ba'])
        BGR = float(request.form['bgr'])
        BU = float(request.form['bu'])
        SC = float(request.form['sc'])
        POT = float(request.form['pot'])
        WC = float(request.form['wc'])
        HTN =float(request.form['htn'])
        DM = float(request.form['dm'])
        CAD = float(request.form['cad'])
        PE = float(request.form['pe'])
        ANE = float(request.form['ane'])

        values = np.array([[Age,BP,AL,SU,RBC,PC,PCC,BA,BGR,BU,SC,POT,WC,HTN,DM,CAD,PE,ANE]])
        prediction = model_kidney.predict(values)
        if prediction == 1:
            res=" It seems like you have a Kidney Disease. Please consult a Doctor. "
        else:
            res= " It seems you are fine. You don't have a Kidney Disease  "

        return render_template('kidneypred.html', prediction=res)


@app.route("/predict5", methods=['POST'])
def predict5():
    if request.method == 'POST':
        Radius_mean = float(request.form['Radius_mean'])
        Texture_mean = float(request.form['Texture_mean'])
        Perimeter_meanL=float(request.form['Perimeter_mean'])
        Area_mean= float(request.form['Area_mean'])
        Smoothness_mean = float(request.form['Smoothness_mean'])
        Compactness_mean = float(request.form['Compactness_mean'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se= float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst =float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])

        fractal_dimension_se = float(request.form['fractal_dimension_se'])

        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])

        concave_points_worst = float(request.form['concave_points_worst'])

        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
       

        values = np.array([[Radius_mean,Texture_mean,Perimeter_meanL,Area_mean,Smoothness_mean,Compactness_mean,concavity_se,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_worst,smoothness_worst,compactness_worst,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])

        prediction = model_cancer.predict(values)
        if prediction == 1:
            res=" It seems like you have a Breast Cancer. Please consult a Doctor. "
        else:
            res= " It seems you are fine. You don't have a Breast Cancer. "

        return render_template('cancer_pred.html', prediction=res)




if __name__ == "__main__":
    app.run(debug=False)
