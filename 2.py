import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageTk


model = load_model('data.h5')

def get_age(distr):
    distr = distr[0] * 4
    if 0.65 <= distr <= 1.4:
        return "0-18"
    elif 1.65 <= distr <= 2.4:
        return "19-30"
    elif 2.65 <= distr <= 3.4:
        return "31-80"
    elif 3.65 <= distr <= 4.4:
        return "80 +"
    return "Unknown"

def get_gender(prob):
    return "Male" if prob[0] < 0.5 else "Female"

def get_result(sample):
    sample = sample / 255
    sample = np.expand_dims(sample, axis=0)
    sample = np.expand_dims(sample, axis=-1)
    val = model.predict(sample)
    age = get_age(val[0])
    gender = get_gender(val[1])
    result_label.config(text=f"Predicted Gender: {gender}\nPredicted Age: {age}")

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)
        show_image(file_path)
        get_result(image)

def show_image(file_path):
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img


root = tk.Tk()
root.title("Gender and Age Prediction")


load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)


root.mainloop()
