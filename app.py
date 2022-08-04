#imports
import os
from flask import Flask
from flask import request,render_template, redirect, url_for
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import cv2

#store the uploaded files and the ALLOWED_EXTENSIONS is the set of allowed file extensions.
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_PATH'] = UPLOAD_FOLDER


saved_model = load_model("./model.h5")

# check if an extension is valid
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route("/", methods=["GET"])
def classify():
    return render_template("index.html")


@app.route("/classify", methods=["GET","POST"])
def index():

    if request.method == "POST":
        file = request.files['filename']


        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename.replace(" ", "_").replace("-", "_8"))
            file.save(file_path)

            img = cv2.imread(file_path)
            img_resize = cv2.resize(img, (224,224))
            img_resize = np.array(img_resize)
            print(img_resize.shape)
            img_resize = img_resize.reshape((-1, 224,224,3))
            preds = saved_model.predict(img_resize)[0]
            label_class = "Benign" if preds.argmax() ==0 else "Malignant"
            label_score = preds.max()* 100
            print(preds)
            preds = label_score
            recommend = label_class
            des_malignant = "Chemo can be used as the main treatment for women whose cancer has spread outside the breast and underarm area to distant organs like the liver or lungs."
            des_benign = "Fine needle aspiration to drain fluid-filled cysts. Surgery to remove lumps (lumpectomy). Oral antibiotics for infections like mastitis.22"
            des = des_benign if label_class == "Benign" else des_malignant
            return render_template("results.html", preds =preds, recommend=recommend, filename=file_path, des=des)

        else:

            return "File type not required"
    print("anything")
    return render_template("classify.html", title = 'Index')

@app.route("/<filename>", methods=["GET"])
def display(filename):
    return redirect(url_for('static', filename = f"./uploads/{filename}"))



if __name__ == "__main__":
    app.run(debug=True)
