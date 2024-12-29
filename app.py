from flask import Flask, request, render_template, url_for
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your model
model = load_model('model.h5')

# Define the class labels
class_labels = ['Healthy', 'Powdery', 'Rusty']

# Define remedies for diseases
disease_remedies = {
    'Powdery': "Apply fungicides containing sulfur or potassium bicarbonate. Ensure proper air circulation around plants. Use a baking soda spray (1 tbsp baking soda, 1/2 tsp liquid soap, 1 gallon water) or neem oil (2 tbsp neem oil in 1 gallon water) weekly. Improve airflow, avoid wet leaves, and remove infected parts promptly.",
    'Rusty': "Remove infected leaves and apply fungicides containing copper. Avoid overhead watering."
}

# Preprocessing function for the uploaded image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Home route to display the upload form and result
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            UPLOAD_FOLDER = os.path.join('static', 'uploads')
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            image = Image.open(file_path)
            processed_image = preprocess_image(image, target_size=(224, 224))

            prediction = model.predict(processed_image)
            predicted_class = class_labels[np.argmax(prediction)]

            remedy = disease_remedies.get(predicted_class, "No remedies needed for healthy leaves.")
            return render_template(
                "upload.html", 
                image_url=url_for('static', filename='uploads/' + file.filename),
                predicted_class=predicted_class,
                remedy=remedy
            )

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
