import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.pyplot import specgram

from librosa import display

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure()
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,1,i)
        librosa.display.waveplot(np.array(f))
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure()
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure()
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def build_features_from_files(sub_dir,file_ext="*.wav"):
    features = []
    for filename in glob.glob(os.path.join(sub_dir, file_ext)):
        # print(fn)
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(filename)
        except Exception as e:
            print("Error encountered while parsing file: ", filename)
            continue
        features.append(np.array(np.append(mfccs.ravel(), np.append(chroma.ravel(), np.append(mel.ravel(), np.append(contrast.ravel(),tonnetz.ravel()))))))
    return np.array(features)

label_map = {'rich': 1, 'notrich': 0}
tr_features_notrich = build_features_from_files('notrich')
y_response_notrich = np.array([label_map['notrich']] * len(tr_features_notrich))
tr_features_notrich = np.concatenate((tr_features_notrich, np.matrix(y_response_notrich).T), axis=1)

tr_features_rich = build_features_from_files('rich')
y_response_rich = np.array([label_map['rich']] * len(tr_features_rich))
tr_features_rich = np.concatenate((tr_features_rich, np.matrix(y_response_rich).T), axis=1)

all_training_data = np.concatenate((tr_features_notrich, tr_features_rich))
np.random.shuffle(all_training_data)

y_train = all_training_data[:,len(all_training_data.T) - 1]
all_training_data = np.delete(all_training_data,-1,1)

y_train = np.concatenate((y_train, (y_train - 1) * -1), axis=1)

y_test = y_train[int(len(y_train) * 0.8):, :]
y_train = y_train[:int(len(y_train) * 0.8), :]

x_test = all_training_data[int(len(all_training_data) * 0.8):, :]
all_training_data = all_training_data[:int(len(all_training_data) * 0.8), :]

def plot_examples():
    sound_file_paths = ["rich/2017-02-20 18_25_41560224.wav","notrich/2017-02-20 17_52_54758949.wav"]
    sound_names = ["rich","not rich"]
    raw_sounds = load_sound_files(sound_file_paths)
    plot_waves(sound_names,raw_sounds)
    plot_specgram(sound_names,raw_sounds)
    plot_log_power_specgram(sound_names,raw_sounds)

# plot_examples()

# configuration parameters required by neural network model
training_epochs = 50
n_dim = all_training_data.shape[1]
n_classes = 2
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

# placeholders for features and class labels, which tensor flow will fill with the data at runtime
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], 
mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.initialize_all_variables()

# The cross-entropy cost function will be minimised using gradient descent optimizer
cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train neural network model, visualise whether cost is decreasing with each epoch and make prediction on the test set
cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:all_training_data,Y:y_train})
        cost_history = np.append(cost_history,cost)
    
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: x_test})
    y_true = sess.run(tf.argmax(y_test,1))
    print("Test accuracy: ",round(sess.run(accuracy, 
    	feed_dict={X: x_test,Y: y_test}),3))

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
print("F-Score:", round(f,3))
