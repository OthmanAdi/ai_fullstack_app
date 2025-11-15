import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

print("MNIST MODEL TRAINING")
print("="*60)
#1 Daten Laden
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Training: {len(X_train)} Bilder") # Text Interpolation

# Normalizierung & Reshapen die Daten = 0-255 → 0.0-1.0
X_train = X_train.astype('float32') / 255.0

X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28*28)
# plt.imshow(X_train[0].reshape(28, 28), cmap="grey")
# plt.title(f"Handgeschrieben Zahl {y_train[0]}")
# plt.show()
X_test = X_test.reshape(-1, 28*28)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("Build Finished")

print("Training...")
model.fit(X_train, y_train, epochs=5, batch_size=157, validation_split=0.1, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

model.save('mnist_model.keras')
print("Model Stored: mnist_model.keras")