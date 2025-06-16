import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("model/mnist_digit_classifier.h5")

# Load data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

# Predict
index = int(input("Enter index (0–9999): "))
img = x_test[index]
prediction = model.predict(np.expand_dims(img, axis=0))
predicted_label = np.argmax(prediction)

# Display result
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {predicted_label} | Actual: {y_test[index]}")
plt.show()
