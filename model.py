# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import h5py
import gdown

# URL of the publicly accessible .h5 file
file_url = "https://drive.google.com/uc?export=download&id=1LKoxag5ORvsiGcDjn_mziNH7tMHk1fDJ"

# Download the file and save it locally as 'model.h5'
output_file = 'lungcancer.h5'
gdown.download(file_url, output_file, quiet=False)

# Load the .h5 file using h5py
model = h5py.File(output_file, 'r')

# Preparing and pre-processing the image
def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((256, 256))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 256, 256, 3)
    return img_reshape


# Predicting function

def predict_result(predict):
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
