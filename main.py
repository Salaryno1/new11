import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def input_dataset():
    mnist = tensorflow.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # print(x_train[0])
    # print(x_train.shape)
    return x_train, y_train, x_test, y_test


def creat_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, [3, 3], activation="relu", input_shape=[28, 28, 1]))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, [3, 3], activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=10, activation="softmax"))
    return model


def one_hot(labels):
    onehot_labels = np.zeros(shape=[len(labels), 10])
    for i in range(len(labels)):
        onehot_labels[i][labels[i]] = 1
    return onehot_labels


def train_model(train_imgs, train_labels, test_imgs, test_labels):
    model_created = creat_model()
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)
    model_created.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss="categorical_crossentropy",
                          metrics=['accuracy'])
    model_created.fit(x=train_imgs, y=train_labels, batch_size=500, epochs=3, validation_split=0.2)

    test_loss, test_acc = model_created.evaluate(x=test_imgs, y=test_labels)
    print("Test Accuracy %.2f" % test_acc)

    # 开始预测
    cnt = 0
    print(len(test_imgs))
    predictions = model_created.predict(test_imgs)
    for i in range(len(test_imgs)):
        target = np.argmax(predictions[i])
        label = np.argmax(test_labels[i])
        if target == label:
            cnt += 1
    print("correct prediction of total : %.2f" % (cnt / len(test_imgs)))
    model_created.save('./model-h5/mnist-model.h5')


if __name__ == '__main__':
    input_x, input_y, input_test_x, input_test_y = input_dataset()
    input_x = np.expand_dims(input_x, -1)
    input_test_x = np.expand_dims(input_test_x, -1)
    print("input_x :{}".format(input_x.shape))
    print("output_labels :{}".format(input_y.shape))
    train_model(input_x, input_y, input_test_x, input_test_y)

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
