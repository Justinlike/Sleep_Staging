import os
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
import tensorflow as tf
import tensorflow.keras.backend as K


config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

import argparse

# from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
# import tensorflow as tf # 用来等价K
from model import create_model

# argument parser
parser = argparse.ArgumentParser(description='Train ResNet-SE-LSTM or dry-run data loading')
parser.add_argument('--data_path', type=str, default='E:/datasets/sleep-edf-database-expanded-1.0.0/sleep-cassette/eeg_fpz_cz',
                    help='Directory containing .npz files with keys x,y (default: sleepedf)')
parser.add_argument('--dry_run', action='store_true', help='Only load and split data, then exit')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--model_read_path', type=str, default='logs/model.keras')
parser.add_argument('--model_save_path', type=str, default='logs', help='Path to saving model (default: none)')
args = parser.parse_args()

## data preparation
data_path = args.data_path

fnames = sorted(glob(os.path.join(data_path, '*.npz')))

X, y = [], []
for fname in fnames:
    samples = np.load(fname)
    X.append(samples['x'])
    y.append(samples['y'])

# one-hot encoding sleep stages    
temp_y = []
for i in range(len(y)):
    temp_ = []
    for j in range(len(y[i])):
        temp = np.zeros((5,))
        temp[y[i][j]] = 1.
        temp_.append(temp)
    temp_y.append(np.array(temp_))
y = temp_y    

# make sequence data
seq_length = 15

X_seq, y_seq = [], []
for i in range(len(X)):
    for j in range(0, len(X[i]), seq_length): # discard last short sequence
        if j+seq_length < len(X[i]):
            X_seq.append(np.array(X[i][j:j+seq_length]))
            y_seq.append(np.array(y[i][j:j+seq_length]))
            
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)
X_seq_train, X_seq_val, y_seq_train, y_seq_val = train_test_split(X_seq_train, y_seq_train, test_size=0.1, random_state=42)

X_seq_train = np.expand_dims(X_seq_train, -1)
X_seq_val = np.expand_dims(X_seq_val, -1)
X_seq_test = np.expand_dims(X_seq_test, -1)

## model training
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='.keras', monitor='val_loss', verbose=1, save_best_only=True)
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
redonplat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
csv_logger = tf.keras.callbacks.CSVLogger('log.csv', separator=',', append=True)
callbacks_list = [
    checkpoint,
    early,
    redonplat,
    csv_logger,
    ]

if args.model_read_path and os.path.exists(args.model_read_path):
    print('Load model from', args.model_path)
    custom_objects = {'tf': tf, 'K': K}
    model = tf.keras.models.load_model('model.keras', custom_objects=custom_objects)
else:
    model = create_model(seq_length=seq_length)


hist = model.fit(X_seq_train, y_seq_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
                 validation_data=(X_seq_val, y_seq_val), callbacks=callbacks_list)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(1,2,2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.suptitle('hist')
plt.savefig('hist.png')
plt.close()

model.save('model.keras')

## output
y_seq_pred = model.predict(X_seq_test, batch_size=1)

y_seq_pred_ = y_seq_pred.reshape(-1,5)
y_seq_test_ = y_seq_test.reshape(-1,5)
y_seq_pred_ = np.array([np.argmax(s) for s in y_seq_pred_])
y_seq_test_ = np.array([np.argmax(s) for s in y_seq_test_])

accuracy = accuracy_score(y_seq_test_, y_seq_pred_)
print('accuracy:', accuracy)

kappa = cohen_kappa_score(y_seq_test_, y_seq_pred_)
print('kappa:', kappa)

label = ['Wake', 'N1', 'N2', 'N3', 'REM']  

report = classification_report(y_true=y_seq_test_, y_pred=y_seq_pred_, target_names=label, output_dict=True)
print('report:', report) 
report = pd.DataFrame(report).transpose()
report.to_csv('report.csv', index= True)

cm = confusion_matrix(y_seq_test_, y_seq_pred_)
sns.heatmap(cm, square=True, annot=True, fmt='d', cmap='YlGnBu', xticklabels=label, yticklabels=label)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig('cm.png', bbox_inches='tight', dpi=300)
plt.close()

cm_norm = confusion_matrix(y_seq_test_, y_seq_pred_, normalize='true')
sns.heatmap(cm_norm, square=True, annot=True, cmap='YlGnBu', xticklabels=label, yticklabels=label)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig('cm_norm.png', bbox_inches='tight', dpi=300)
plt.close()

    