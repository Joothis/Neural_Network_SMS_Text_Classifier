# 📩 SMS Spam Classifier

This project is a **Neural Network-based SMS Spam Classifier** that uses **TF-IDF vectorization** and **Deep Learning** to classify messages as spam or not.

## 📌 Features

- Uses **TF-IDF** to convert text into numerical features.
- Implements a **Neural Network** for classification.
- Achieves high accuracy in detecting spam messages.
- Saves and loads the trained model for future predictions.

## 🚀 Installation

 **Clone the repository**
   ```sh
   !git clone https://github.com/joothis/Neural_Network_SMS_Text_Classifier.git
   cd sms-spam-classifier
   ```


## 📊 Dataset

The model uses a dataset containing SMS messages labeled as spam or ham. Ensure your dataset (`spam.csv`) is in the project directory before running the script.

## 🏗️ Model Training

- The script trains the model using **TF-IDF** and a **simple feedforward neural network**.
- The best model is saved as `sms_spam_model.h5`.

## 🔍 Making Predictions

You can use the trained model to predict new messages:

```python
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Load model
model = load_model('sms_spam_model.h5')
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Predict
message = ["Congratulations! You won a free iPhone"]
message_tfidf = vectorizer.transform(message).toarray()
prediction = model.predict(message_tfidf)
print("Spam" if prediction > 0.5 else "Not Spam")
```



## 📜 License

This project is open-source under the **MIT License**.

---

🔹 **Author**: Joothiswaran Palanisamy\
📧 **Email**: [joothiswaranpalanisamy2005@gmail.com](mailto\:joothiswaranpalanisamy2005@gmail.com)

