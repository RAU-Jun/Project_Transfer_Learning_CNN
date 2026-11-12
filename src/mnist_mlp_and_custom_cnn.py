# 필요한 모듈 임포트
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------
# MNIST 데이터셋을 활용한 간단한 신경망 모델 학습
# -----------------------------------------

#MNIST 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터셋 로드

#데이터 정규화 (0~1 범위)
x_train = x_train / 255.0
x_test = x_test / 255.0

#MNIST 모델 정의
mnist_model = Sequential([
    Flatten(input_shape=(28, 28)),  # 입력 이미지를 1차원 벡터로 변환
    Dense(128, activation='relu'),  # 은닉층
    Dense(10, activation='softmax')  # 출력층 (10개의 클래스)
    ])

#MNIST 모델 컴파일
learning_rate = 0.0001
mnist_model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

#MNIST 모델 학습
start_time = time.monotonic()  # 학습 시간 측정 시작

history = mnist_model.fit(
    x_train, tf.one_hot(y_train, 10),  # 라벨을 원-핫 인코딩
    epochs=200,
    batch_size=32,
    validation_data=(x_test, tf.one_hot(y_test, 10))
    )

end_time = time.monotonic()  # 학습 시간 측정 종료
delta_time = end_time - start_time
print(f"MNIST model learning time: {delta_time:.2f}초")

#MNIST 모델 평가
loss, accuracy = mnist_model.evaluate(x_test, tf.one_hot(y_test, 10))
print(f'MNIST test loss: {loss:.4f}')
print(f'MNIST test accuracy: {accuracy:.4f}')

#MNIST 학습 결과 시각화
#손실 그래프 그리기
plt.plot(history.history['loss'], marker='o', label='tran loss')
plt.plot(history.history['val_loss'], marker='x', label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('MNIST training and val loss')
plt.show()

#정확도 그래프 그리기
plt.plot(history.history['accuracy'], marker='o', label='train accuracy')
plt.plot(history.history['val_accuracy'], marker='x', label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('MNIST training and val accuracy')
plt.show()

#Google Drive 마운트 및 데이터 경로 설정
drive.mount('/content/drive')  #Google Drive를 마운트하여 데이터에 접근
#이미지 데이터 폴더 경로 설정
image_folder_path = '/content/drive/MyDrive/VisionCNNdata'

#폴더가 없으면 생성
if not os.path.exists(image_folder_path):
  os.makedirs(image_folder_path)
  print(f"{image_folder_path} 폴더가 생성되었습니다.")
else:
  print(f"{image_folder_path} 폴더가 이미 존재합니다.")

#데이터 증강 설정 및 데이터 로드
datagen = ImageDataGenerator(
    rotation_range=10,       #이미지 회전 범위 (최대 10도)
    width_shift_range=0.1,   #가로 이동 범위 (이미지 너비의 10%)
    height_shift_range=0.1,  #세로 이동 범위 (이미지 높이의 10%)
    zoom_range=0.1,          #확대/축소 범위 (최대 10%)
    rescale=1.0/255.0,       #픽셀 값을 0~1로 정규화
    validation_split=0.2     #데이터의 20%를 검증 세트로 분할
    )

#학습 데이터 로드
train_data = datagen.flow_from_directory(
    image_folder_path,
    target_size=(28, 28),     #이미지 크기를 28x28로 조정
    color_mode='grayscale',   #흑백 이미지로 로드
    batch_size=32,
    class_mode='categorical',  #다중 클래스 분류
    subset='training'          #학습 데이터로 사용
    )

#검증 데이터 로드
val_data = datagen.flow_from_directory(
    image_folder_path,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'        #검증 데이터로 사용
    )
#CNN 모델 정의
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  #첫 번째 합성곱 층
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),  #두 번째 합성곱 층
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),  #세 번째 합성곱 층
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),  #평탄화
    Dense(128, activation='relu'),  #완전 연결층
    Dropout(0.5),  #과적합 방지를 위한 드롭아웃
    Dense(train_data.num_classes, activation='softmax')  #출력층
    ])

#특정 레이어 고정 (전이 학습)
for layer in cnn_model.layers[:-3]:
  layer.trainable = False  #마지막 세 개의 레이어를 제외한 나머지 레이어의 가중치를 고정

#CNN 모델 컴파일
learning_rate = 0.0001
cnn_model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

#CNN 모델 학습
if train_data.samples == 0 or val_data.samples == 0:
  print("학습 또는 검증 데이터가 없습니다. image_folder_path를 확인하세요.")
  cnn_history = None
else:
  cnn_history = cnn_model.fit(
      train_data,
      epochs=200,
      validation_data=val_data
      )

#CNN 모델 평가
steps = val_data.samples // val_data.batch_size
if steps == 0:
  print("검증 데이터에 배치가 없습니다. 평가를 건너뜁니다.")
else:
  loss, accuracy = cnn_model.evaluate(val_data, steps=steps)
  print(f'CNN test loss: {loss:.4f}')
  print(f'CNN test accuracy: {accuracy:.4f}')

#CNN 학습 결과 시각화
if cnn_history is not None:
  #손실 그래프 그리기
  plt.plot(cnn_history.history['loss'], marker='o', label='train loss')
  plt.plot(cnn_history.history['val_loss'], marker='x', label='val loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.title('CNN training and val loss')
  plt.show()

  #정확도 그래프 그리기
  plt.plot(cnn_history.history['accuracy'], marker='o', label='train accuracy')
  plt.plot(cnn_history.history['val_accuracy'], marker='x', label='val accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.title('CNN training and val accuracy')
  plt.show()
else:
  print("학습 또는 검증 데이터가 없습니다. image_folder_path를 확인하세요.")
