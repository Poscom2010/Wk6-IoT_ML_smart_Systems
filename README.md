# Wk6-IoT_ML_smart_Systems

# 🗑️ Garbage Classification using CNN | Week 6 IoT + ML Smart Systems

This project is part of the **PLP Academy Week 6 Assignment** under the module **IoT and Machine Learning Smart Systems**.  
The goal is to train a deep learning model that classifies images of garbage into various categories using Convolutional Neural Networks (CNNs) with TensorFlow and Keras.

---

## 📁 Dataset

We used a **garbage classification dataset** structured in the following format:

garbage_dataset/
├── train/
│ ├── cardboard/
│ ├── glass/
│ ├── metal/
│ ├── paper/
│ ├── plastic/
│ └── trash/
└── valid/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/

yaml
Copy
Edit

The dataset is loaded using `ImageDataGenerator` for preprocessing and batching.

---

## 🧠 Model Architecture

We built a custom **Convolutional Neural Network (CNN)** with:

- Input layer: 150x150 RGB images
- Multiple Conv2D + MaxPooling2D layers
- Dropout layers to reduce overfitting
- Fully connected `Dense` layers
- Output layer with `softmax` activation for **categorical classification**

---

## 🔧 Technologies Used

- **TensorFlow / Keras**
- **Python**
- **Google Colab**
- **Git & GitHub**

---

## 📊 Training Summary

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Epochs**: Customizable
- **Validation Split**: Handled using a separate validation folder

---

## 📁 Project Structure

.
├── garbage_classifier_model.h5 # Final trained CNN model
├── Garbage_Classification.ipynb # Jupyter notebook with full code
├── README.md # Project documentation
└── dataset/ # (Optional) Garbage dataset used for training

yaml
Copy
Edit

---

## 🚀 How to Run

### 1. Clone the repo

##```bash
git clone https://github.com/Markayo21/Wk6-IoT_ML_smart_Systems.git
cd Wk6-IoT_ML_smart_Systems
2. Open the Notebook
Open Garbage_Classification.ipynb in Google Colab or Jupyter Notebook and follow the steps.

3. Load the Model
If you just want to use the trained model:

python
Copy
Edit
from tensorflow.keras.models import load_model
model = load_model("garbage_classifier_model.h5")
4. Make Predictions
Upload an image and preprocess it as follows:

python
Copy
Edit
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('your_image.jpg', target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction[0])```



