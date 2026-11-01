import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tensorflow import keras
from tensorflow.keras import layers
from common import load_data

os.makedirs('outputs', exist_ok=True)
X_train, y_train, X_test, y_test, class_names, labels = load_data()
Xf_train = X_train.reshape(len(X_train), -1)   # 32 x 32 = 1024 features per image
Xf_test = X_test.reshape(len(X_test), -1)

# ---- 1. PCA: how many components do we need? ----
pca_full = PCA().fit(Xf_train)
cum = np.cumsum(pca_full.explained_variance_ratio_)
print('Components needed for 95% of the variance:', int(np.argmax(cum >= 0.95)) + 1)
plt.figure()
plt.plot(cum)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.savefig('outputs/pca_variance.png', dpi=150)
plt.close()

# ---- 2. The first 16 principal components shown as images ----
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, comp in zip(axes.ravel(), pca_full.components_[:16]):
    ax.imshow(comp.reshape(32, 32), cmap='gray')
    ax.axis('off')
plt.savefig('outputs/eigen_characters.png', dpi=150)
plt.close()

# ---- 3. Reconstruction with 20, 50 and 100 components ----
samples = Xf_test[np.linspace(0, len(Xf_test) - 1, 8).astype(int)]
rows = [('original', samples)]
for k in (20, 50, 100):
    p = PCA(k).fit(Xf_train)
    rows.append((str(k) + ' components', p.inverse_transform(p.transform(samples))))
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for r, (title, imgs) in enumerate(rows):
    for c in range(8):
        axes[r, c].imshow(imgs[c].reshape(32, 32), cmap='gray')
        axes[r, c].axis('off')
    axes[r, 0].set_title(title, fontsize=8, loc='left')
plt.savefig('outputs/reconstruction.png', dpi=150)
plt.close()

# ---- 4. PCA vs LDA in 2D (5000 random training images) ----
# LDA may print a 'Variables are collinear' warning. That is harmless here.
lda = LDA(n_components=45).fit(Xf_train, y_train)   # at most classes - 1 = 45
idx = np.random.RandomState(0).choice(len(Xf_train), 5000, replace=False)
z_pca = PCA(2).fit(Xf_train).transform(Xf_train[idx])
z_lda = lda.transform(Xf_train[idx])[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(z_pca[:, 0], z_pca[:, 1], c=y_train[idx], cmap='nipy_spectral', s=4)
axes[0].set_title('PCA: first 2 components')
axes[1].scatter(z_lda[:, 0], z_lda[:, 1], c=y_train[idx], cmap='nipy_spectral', s=4)
axes[1].set_title('LDA: first 2 components')
plt.savefig('outputs/pca_vs_lda.png', dpi=150)
plt.close()

# ---- 5. Same MLP on raw pixels vs PCA features vs LDA features ----
def make_mlp(input_dim):
    m = keras.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(256, activation='relu'), layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(46, activation='softmax')])
    m.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

pca = PCA(100).fit(Xf_train)
inputs = {
    'Raw pixels (1024)': (Xf_train, Xf_test),
    'PCA (100)': (pca.transform(Xf_train), pca.transform(Xf_test)),
    'LDA (45)': (lda.transform(Xf_train), lda.transform(Xf_test)),
}
results = []
for name, (tr, te) in inputs.items():
    m = make_mlp(tr.shape[1])
    start = time.time()
    m.fit(tr, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=0)
    acc = m.evaluate(te, y_test, verbose=0)[1]
    line = f'{name:20s} test accuracy = {acc:.4f}   training time = {time.time() - start:.0f}s'
    print(line)
    results.append(line)
with open('outputs/mlp_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print('Done. Check the outputs/ folder.')