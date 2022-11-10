import os
import tempfile

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_addons as tfa

from clearml import Task, Dataset
from tensorflow import keras
from keras.layers import Input, Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint

# adds requirements for the agents to run in pip mode.
# What is an agent? Any machine we set up to execute Tasks.
Task.add_requirements('requirements.txt')
task = Task.init(project_name = 'ClearML-demo', task_name = "MNIST classifier", tags=['Script'], task_type='training')

# task.execute_remotely("GPU")

task_params = {
    'batch_size' : 32,
    'epochs' : 3,
    'hidden_dims' : (512, 256),
    'dropout' : 0.3
}
task.connect(task_params)

def main():
    # You can pass them the dataset_id if you want to target a specific one. 
    # Otherwise it will get the latest one if there is more than one match in the search.
    ds = Dataset.get(
        dataset_project = 'ClearML-demo',
        dataset_name = "MNIST"
    )

    # The dataset is not downloaded. This is the metadata.
    ds.list_files()

    # A mutable local copy:
    data_path = ds.get_mutable_local_copy(target_folder = './Data', overwrite=True)

    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    # Model definition
    model = keras.Sequential([Input((784,))])

    for h_dim in task_params['hidden_dims']:
        model.add(Dense(h_dim))
        model.add(Activation(tfa.activations.mish))
        model.add(Dropout(task_params['dropout']))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # Callbacks
    tb_callback = TensorBoard(log_dir = tempfile.gettempdir())
    model_store = ModelCheckpoint(filepath=os.path.join(tempfile.gettempdir(), 'weight.{epoch}.hdf5'))

    class ConfMatCallback(keras.callbacks.Callback):
        def __init__(self, X_test, y_test):
            self.X_test = X_test[:1000]
            self.y_test = y_test[:1000]

        def on_epoch_end(self, epoch, logs=None):
            pred_classes = np.argmax(self.model.predict(self.X_test),axis=1)
            true_classes = np.argmax(self.y_test, axis=1)
            conf_mat = tf.math.confusion_matrix(labels=true_classes, predictions=pred_classes).numpy()
            figure = plt.figure(figsize=(10, 10))
            plt.ioff()
            sns.heatmap(conf_mat, annot=True, cmap=plt.cm.Blues)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            task.logger.report_matplotlib_figure(title="Confusion Matrix", series = "", iteration=epoch, figure=plt)
            plt.close()
                                    
    conf_mat_callback = ConfMatCallback(X_test, y_test)

    # Fit the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train,
            batch_size = task_params['batch_size'], 
            epochs = task_params['epochs'],
            callbacks = [tb_callback, model_store, conf_mat_callback],
            validation_data = (X_test, y_test)
            )

    # Evaluate the model and log the result to ClearML
    score = model.evaluate(X_test, y_test)
    task.logger.report_single_value("Model final score", value=score[0])
    task.logger.report_single_value("Model final score", value=score[1])

if __name__ == '__main__':
    main()

# Why all this?
# Let's launch the clearml_agent notebook. Of course you can just call clearml-init on any machine you ssh as wel.