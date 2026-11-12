from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from os import system
from matplotlib import pyplot as plt

model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
MaxPooling2D(pool_size=(2, 2)),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(128, activation='relu'),
Dense(10, activation='softmax')
])

#Conv2D, Maxpooling2D 동결
for layer in model.layers[:-2]:
  layer.trainable = False

learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=learning_rate),
loss='categorical_crossentropy',
metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train /255.0

x_test = x_test / 255.0

history = model.fit(x_train, tf.one_hot(y_train, 10),
epochs=200,
batch_size=32,
validation_data=(x_test, tf.one_hot(y_test, 10)))

loss, accuracy = model.evaluate(x_test, tf.one_hot(y_test, 10))
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='x', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='x', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

system("Pause")
