from collections import Counter
import time

from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LSTM,
    MaxPooling2D,
    Reshape
)
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class NeuralNetwork:
    def train(self):
        start_time = time.perf_counter()
        checkpoint = ModelCheckpoint(
            self.weight_filepath,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        callbacks_list = [checkpoint]
        self.model.fit(
            self.train_vectors,
            self.train_labels,
            epochs=20,
            batch_size=64,
            shuffle=False,
            validation_data=(self.test_vectors, self.test_labels),
            callbacks=callbacks_list
        )

        self.train_time = (time.perf_counter() - start_time)

    def test(self):
        start_time = time.perf_counter()
        self.model.load_weights(self.weight_filepath)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        pred = self.model.predict(self.test_vectors, batch_size=64)
        pred_labels = np.argmax(pred, axis=1)

        self.metrics = self.calculate_metrics(pred_labels)
        self.confusion = confusion_matrix(self.real_labels, pred_labels)
        self.test_time = (time.perf_counter() - start_time)
    
    def calculate_prediction_time(self, pred_input):
        start_time = time.perf_counter()
        pred = self.model.predict(pred_input, batch_size=1)
        self.prediction_time = (time.perf_counter() - start_time)
        pred_labels = np.argmax(pred, axis=1)
        print(pred_labels, self.prediction_time)
        return self.prediction_time

    def calculate_metrics(self, pred_labels):
        result = {}
        if self.qtd_classes == 2:
            result = self.evaluate_class(pred_labels, 1)
            result['type'] = 'binary'
        else:
            result['type'] = 'multiclass'
            for i in range(self.qtd_classes):
                result[f'class_{i}'] = self.evaluate_class(pred_labels, i)
        return result

    def evaluate_class(self, pred_labels, class_id):
        result_prediction = []
        for pred, real in np.nditer([pred_labels, self.real_labels]):
            if pred == class_id and real == class_id:
                result_prediction.append('True positive')
            elif pred == class_id:
                result_prediction.append('False positive')
            elif real == class_id:
                result_prediction.append('False negative')
            else:
                result_prediction.append('True negative')
        result_count = Counter(result_prediction)
        true_positives = result_count['True positive']
        true_negatives = result_count['True negative']
        positives = true_positives + result_count['False positive']
        relevants = true_positives + result_count['False negative']

        accuracy = (
            true_positives + true_negatives
        )/len(result_prediction)
        precision = true_positives / positives
        recall = true_positives / relevants
        f1 = 2*((precision * recall)/(precision + recall))

        return {
            'vectorizing_time': self.vectorizing_time,
            'train_time': self.train_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class ConvolutedNeuralNetwork(NeuralNetwork):
    def __init__(
            self,
            train_vectors,
            train_labels,
            test_vectors,
            test_labels,
            weight_filepath,
            vectorizing_time,
            optimizer_alg='ADAM'
    ):
        NeuralNetwork.__init__(self)
        start_time = time.perf_counter()

        self.vectorizing_time = vectorizing_time

        self.loss_function = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.train_time = 0
        self.metrics_raw = None
        self.confusion_raw = None
        self.test_raw_time = 0
        self.metrics_load = None
        self.confusion_load = None
        self.test_load_time = 0

        self.weight_filepath = weight_filepath

        self.qtd_classes = len(list(set(train_labels)))
        self.train_vectors = train_vectors
        self.train_labels = to_categorical(
            train_labels,
            num_classes=self.qtd_classes
        )
        self.test_vectors = test_vectors
        self.test_labels = to_categorical(
            test_labels,
            num_classes=self.qtd_classes
        )
        self.real_labels = np.array([int(item) for item in test_labels])

        self.vector_length = len(train_vectors[0, :, 0])
        self.vector_dimension = len(train_vectors[0, 0, :])

        self.model = Sequential()
        self.model.add(
            Reshape(
                (self.vector_length, self.vector_dimension, 1),
                input_shape=(self.vector_length, self.vector_dimension)
            )
        )
        self.model.add(
            Conv2D(
                200,
                (2, self.vector_dimension),
                strides=(1, 1),
                padding='valid',
                activation='relu',
                use_bias=True
            )
        )
        output = self.model.output_shape
        self.model.add(MaxPooling2D(pool_size=(output[1], output[2])))
        self.model.add(Dropout(.5))
        self.model.add(Flatten())
        self.model.add(Dense(self.qtd_classes, activation='softmax'))
        if optimizer_alg == 'SGD':
            self.optimizer = SGD()
        else:  # default optimizer is ADAM
            self.optimizer = Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08
            )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        self.load_time = time.perf_counter() - start_time


class LongShortTermMemoryNetwork(NeuralNetwork):
    def __init__(
        self,
        train_vectors,
        train_labels,
        test_vectors,
        test_labels,
        weight_filepath,
        vectorizing_time,
        optimizer_alg='ADAM'
    ):
        start_time = time.perf_counter()

        self.vectorizing_time = vectorizing_time

        self.loss_function = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.train_time = 0
        self.metrics_raw = None
        self.confusion_raw = None
        self.test_raw_time = 0
        self.metrics_load = None
        self.confusion_load = None
        self.test_load_time = 0

        self.weight_filepath = weight_filepath

        self.qtd_classes = len(list(set(train_labels)))
        self.train_vectors = train_vectors
        self.train_labels = to_categorical(
            train_labels,
            num_classes=self.qtd_classes
        )
        self.test_vectors = test_vectors
        self.test_labels = to_categorical(
            test_labels,
            num_classes=self.qtd_classes
        )
        self.real_labels = np.array([int(item) for item in test_labels])

        self.vector_length = len(train_vectors[0, :, 0])
        self.vector_dimension = len(train_vectors[0, 0, :])

        self.model = Sequential()
        self.model.add(
            LSTM(
                50,
                return_sequences=False,
                input_shape=(self.vector_length, self.vector_dimension)
            )
        )
        self.model.add(Dropout(.50))
        self.model.add(
            Dense(
                self.qtd_classes,
                activation='softmax',
                kernel_regularizer=regularizers.l2(0.01)
            )
        )

        if optimizer_alg == 'SGD':
            self.optimizer = SGD()
        else:  # default optimizer is ADAM
            self.optimizer = Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08
            )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        self.load_time = time.perf_counter() - start_time


class CombinedNeuralNetworks(NeuralNetwork):
    def __init__(
        self,
        train_vectors,
        train_labels,
        test_vectors,
        test_labels,
        weight_filepath,
        vectorizing_time,
        optimizer_alg='ADAM'
    ):
        start_time = time.perf_counter()

        self.vectorizing_time = vectorizing_time

        self.loss_function = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.train_time = 0
        self.metrics_raw = None
        self.confusion_raw = None
        self.test_raw_time = 0
        self.metrics_load = None
        self.confusion_load = None
        self.test_load_time = 0

        self.weight_filepath = weight_filepath

        self.qtd_classes = len(list(set(train_labels)))
        self.train_vectors = train_vectors
        self.train_labels = to_categorical(
            train_labels,
            num_classes=self.qtd_classes
        )
        self.test_vectors = test_vectors
        self.test_labels = to_categorical(
            test_labels,
            num_classes=self.qtd_classes
        )
        self.real_labels = np.array([int(item) for item in test_labels])

        self.vector_length = len(train_vectors[0, :, 0])
        self.vector_dimension = len(train_vectors[0, 0, :])

        self.model = Sequential()
        self.model.add(
            Reshape(
                (self.vector_length, self.vector_dimension, 1),
                input_shape=(self.vector_length, self.vector_dimension)
            )
        )
        self.model.add(
            Conv2D(
                200,
                (2, self.vector_dimension),
                strides=(1, 1),
                padding='valid',
                activation='relu',
                use_bias=True
            )
        )
        output = self.model.output_shape
        self.model.add(Reshape((output[1], output[3])))
        self.model.add(Dropout(.25))
        self.model.add(
            LSTM(
                100,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='hard_sigmoid'
            )
        )
        self.model.add(Dropout(.50))
        self.model.add(
            Dense(
                self.qtd_classes,
                activation='softmax',
                kernel_regularizer=regularizers.l2(0.01)
            )
        )

        if optimizer_alg == 'SGD':
            self.optimizer = SGD()
        else:  # default optimizer is ADAM
            self.optimizer = Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08
            )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        self.load_time = time.perf_counter() - start_time
