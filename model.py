import tensorflow as tf
import numpy as np
import requests
import io

# Загрузка данных из Google
def get_data_from_google(query):
    response = requests.get(f'https://www.google.com/search?q={query}&tbm=isch')
    html = response.text
    image_urls = re.findall('"ou":"(.*?)"', html)
    images = []
    for url in image_urls:
        response = requests.get(url)
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        images.append(image)
    return images

# Создание модели
def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Обучение модели
def train_model(model, images):
    model.fit(images, epochs=10)

# Запуск модели
if __name__ == '__main__':
    query = 'google.com'
    images = get_data_from_google(query)
    model = create_model()
    train_model(model, images)
