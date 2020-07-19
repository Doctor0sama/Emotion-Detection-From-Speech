import os
import librosa
import numpy as np
import csv

path = "Waves"
lst = []
ft=[]
label=[]

for subdir, dirs, files in os.walk(path):
    for file in files:
        try:
            X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=16000, n_mfcc=20).T, axis=0)
            file = file[1:2]

            arr = mfccs, file
            lst.append(arr)
            ft.append(mfccs)
            label.append(file)
        except ValueError:
            continue

path = 'Khaled Dataset'
for path, _, files in os.walk(path):
    for file in files:
        try:
            X, sample_rate = librosa.load(os.path.join(path, file), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=16000, n_mfcc=20).T, axis=0)
            name = file[4:5]
            if name == "h":
                name = name.capitalize()
            elif name.startswith('a'):
                name = 'S'

            arr = mfccs, name
            lst.append(arr)
            ft.append(mfccs)
            label.append(name)
        except ValueError:
            continue

X, y = zip(*lst)

with open('features.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(ft)

with open('labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(label)
