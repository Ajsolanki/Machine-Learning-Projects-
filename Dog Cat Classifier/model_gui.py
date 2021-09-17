import tkinter as tk
import tensorflow as tf
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

## Creating a TKinter Window 
root = tk.Tk()
root.title('Image Classifier')
root.iconbitmap('class.ico')
root.resizable(False, False)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

img_width, img_height = 128, 128
img_channels = 3
img_size = (img_width, img_height)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, img_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because two classes Dog & Cat

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
## Priting Model Summary 
# model.summary()
model.load_weights("C:\/Users\/assol\Documents\Python Scripts\/NMG image_classifier\model(30).h5")


'''
To associate action with GUI button to perform an action, when a user interact with a GUI button. Define functions 
'''

## Function to load Image and showing in the window 
def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="C:\/Users\/assol\Documents\/Python Scripts\/NMG image_classifier\/test1", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 400 # Processing image for displaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()
    
    
def classify():
    """
        Function to call Trained model and processing on image. After processing the image,
        Showing model predications and associated confidence percentage

    """
    img = tf.io.read_file(image_data)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img, [128,128])
    img = tf.expand_dims(img,axis=0)
    predictions = model.predict(img)    
    label = predictions[0] 
    table = tk.Label(frame, text="Image class Predictions and Confidences percentage").pack()
    result = tk.Label(frame, 
                      text = 'Cat :' + str(round(float(label[0])*100,3)) + '%').pack()
    result = tk.Label(frame, 
                      text = 'Dog :' + str(round(float(label[1])*100,3)) + '%').pack()
    
         
         
## Load Model and weights from saved model

    
## Add Title of the window 
tit = tk.Label(root, text="Image Classifier", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=800, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.9, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Select Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)

## Load Saved model
model = model()
root.mainloop()

