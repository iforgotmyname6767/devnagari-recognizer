import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # save plots as files instead of opening windows
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from common import load_data

os.makedirs('outputs', exist_ok=True)
X_train, y_train, X_test, y_test, class_names, labels = load_data()
print('Train:', X_train.shape, 'Test:', X_test.shape, 'Classes:', len(class_names))

# ---- 1. Build and train the CNN ----
cnn = keras.Sequential([
    layers.Input((32, 32, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dense(46, activation='softmax')])
cnn.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

hist = cnn.fit(X_train, y_train, epochs=12, batch_size=128, validation_split=0.1,
               callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
test_acc = cnn.evaluate(X_test, y_test, verbose=0)[1]
print('CNN test accuracy:', round(float(test_acc), 4))
cnn.save('model/devanagari_cnn.keras')

# ---- 2. Architecture summary and training curves (for your slides) ----
lines = []
cnn.summary(print_fn=lines.append)
with open('outputs/cnn_summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist.history['loss'], label='train')
axes[0].plot(hist.history['val_loss'], label='validation')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(hist.history['accuracy'], label='train')
axes[1].plot(hist.history['val_accuracy'], label='validation')
axes[1].set_title('Accuracy'); axes[1].legend()
plt.savefig('outputs/cnn_curves.png', dpi=150)
plt.close()

# ---- 3. Evaluation: report and most-confused character pairs ----
y_pred = cnn.predict(X_test, verbose=0).argmax(axis=1)
with open('outputs/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
np.fill_diagonal(cm, 0)
pairs = []
for idx in np.argsort(cm.ravel())[::-1][:10]:
    i, j = divmod(idx, cm.shape[1])
    pairs.append(f'{labels[i]} ({class_names[i]}) predicted as {labels[j]} ({class_names[j]}): {cm[i, j]} times')
with open('outputs/most_confused.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(pairs))

# ---- 4. Convert to TFLite for deployment and check that accuracy survived ----
converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
with open('model/devanagari_cnn.tflite', 'wb') as f:
    f.write(converter.convert())

interp = tf.lite.Interpreter(model_path='model/devanagari_cnn.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]['index']
out = interp.get_output_details()[0]['index']
sample = np.random.RandomState(0).choice(len(X_test), 500, replace=False)
correct = 0
for i in sample:
    interp.set_tensor(inp, X_test[i:i + 1])
    interp.invoke()
    correct += int(interp.get_tensor(out).argmax() == y_test[i])
print('TFLite accuracy on 500 random test images:', correct / 500)
print('Done. Check the model/ and outputs/ folders.')