
from keras.models import load_model
import numpy as np
import cv2


model = load_model('data.h5')

def get_age(distr):
    distr = distr[0]*4
    if distr >= 0.65 and distr <= 1.4: return "0-18"
    if distr >= 1.65 and distr <= 2.4: return "19-30"
    if distr >= 2.65 and distr <= 3.4: return "31-80"
    if distr >= 3.65 and distr <= 4.4: return "80 +"
    return "Unknown"

def get_gender(prob):
    if prob[0] < 0.5: return "Male"
    else: return "Female"

def get_result(sample):
    sample = sample/255
    sample = np.expand_dims(sample, axis=0) 
    sample = np.expand_dims(sample, axis=-1)  
    val = model.predict(sample)
    age = get_age(val[0])
    gender = get_gender(val[1])
    print("Values:",val,"\nPredicted Gender:",gender,"Predicted Age:",age)


image = cv2.imread(r"C:\Users\gurud\OneDrive\Desktop\Tom_Holland_by_Gage_Skidmore.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (64, 64))  
image = image / 255.0  
image = np.expand_dims(image, axis=-1)  

get_result(image)
