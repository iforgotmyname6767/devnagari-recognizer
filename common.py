import json
import os
import numpy as np
import tensorflow as tf

CACHE = 'data/cache.npz'

# Devanagari characters in the same order as the folders character_1 ... character_36
CONS = 'क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह क्ष त्र ज्ञ'.split()
DIGITS = '०१२३४५६७८९'


def pretty(folder_name):
    parts = folder_name.split('_')
    if parts[0] == 'character':
        return CONS[int(parts[1]) - 1]
    return DIGITS[int(parts[1])]


def _load_split(path):
    ds = tf.keras.utils.image_dataset_from_directory(
        path, color_mode='grayscale', image_size=(32, 32),
        batch_size=256, shuffle=False)
    X = np.concatenate([x.numpy() for x, _ in ds]).astype('uint8')
    y = np.concatenate([y.numpy() for _, y in ds])
    return X, y, ds.class_names


def load_data():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
        class_names = [str(n) for n in d['class_names']]
    else:
        Xtr, ytr, class_names = _load_split('data/Train')
        Xte, yte, _ = _load_split('data/Test')
        np.savez(CACHE, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, class_names=np.array(class_names))

    # The images load sorted by class, so shuffle the training set once.
    # Without this, the validation split would contain only the last few classes.
    p = np.random.RandomState(42).permutation(len(Xtr))
    Xtr, ytr = Xtr[p], ytr[p]

    labels = [pretty(n) for n in class_names]
    os.makedirs('model', exist_ok=True)
    with open('model/labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False)

    return (Xtr.astype('float32') / 255.0, ytr,
            Xte.astype('float32') / 255.0, yte, class_names, labels)