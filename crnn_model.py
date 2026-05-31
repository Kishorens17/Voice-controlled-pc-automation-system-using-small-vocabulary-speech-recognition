"""
CRNN (CNN + BiGRU + Attention) for Speech Command Recognition.

Anti-overfitting measures:
  - Lightweight architecture (~80K params)
  - SpecAugment (on-the-fly frequency/time masking via callback)
  - Batch Normalization + Dropout
  - L2 weight regularization
  - Label smoothing (0.1)
  - Early stopping (patience=20)
  - ReduceLR on plateau (patience=8)
  - Class-weighted loss via sample_weight
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================== Config ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "processed_data.npz")
MODEL_FILE = os.path.join(BASE_DIR, "speech_crnn.keras")

EPOCHS = 60
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.1
L2_REG = 1e-4
LEARNING_RATE = 5e-4


# ======================== SpecAugment as Callback ========================
class SpecAugmentCallback(keras.callbacks.Callback):
    """Apply SpecAugment to training data at the start of each epoch."""

    def __init__(self, X_train_original, freq_mask=8, time_mask=10,
                 n_freq=2, n_time=2):
        super().__init__()
        self.X_original = X_train_original.copy()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time

    def on_epoch_begin(self, epoch, logs=None):
        """Create a new augmented copy of training data each epoch."""
        augmented = self.X_original.copy()
        for i in range(len(augmented)):
            augmented[i, ..., 0] = self._augment(augmented[i, ..., 0])
        # Update the training data in-place
        self.model._train_X[:] = augmented

    def _augment(self, mel):
        aug = mel.copy()
        n_mels, n_steps = aug.shape
        for _ in range(self.n_freq):
            f = np.random.randint(0, min(self.freq_mask, n_mels))
            f0 = np.random.randint(0, max(1, n_mels - f))
            aug[f0:f0+f, :] = 0.0
        for _ in range(self.n_time):
            t = np.random.randint(0, min(self.time_mask, n_steps))
            t0 = np.random.randint(0, max(1, n_steps - t))
            aug[:, t0:t0+t] = 0.0
        return aug


def apply_spec_augment(X, freq_mask=8, time_mask=10, n_freq=2, n_time=2):
    """Apply SpecAugment to a batch of spectrograms."""
    X_aug = X.copy()
    for i in range(len(X_aug)):
        mel = X_aug[i, ..., 0]
        n_mels, n_steps = mel.shape
        for _ in range(n_freq):
            f = np.random.randint(0, min(freq_mask, n_mels))
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[f0:f0+f, :] = 0.0
        for _ in range(n_time):
            t = np.random.randint(0, min(time_mask, n_steps))
            t0 = np.random.randint(0, max(1, n_steps - t))
            mel[:, t0:t0+t] = 0.0
        X_aug[i, ..., 0] = mel
    return X_aug


# ======================== Attention Layer ========================
class Attention(layers.Layer):
    """Additive attention over time steps."""

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_w', shape=(input_shape[-1], 1),
            initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(
            name='att_b', shape=(input_shape[1], 1),
            initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()


# ======================== Model ========================
def build_model(input_shape, num_classes):
    """Build a lightweight CRNN with attention (~80K params)."""
    reg = regularizers.l2(L2_REG)
    inputs = layers.Input(shape=input_shape)

    # --- CNN blocks ---
    x = layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # --- Reshape for RNN ---
    shape = keras.backend.int_shape(x)       # (None, freq', time', 64)
    x = layers.Permute((2, 1, 3))(x)         # (None, time', freq', 64)
    x = layers.Reshape((-1, shape[1] * shape[3]))(x)

    # Reduce feature dim before RNN
    x = layers.TimeDistributed(
        layers.Dense(64, activation='relu', kernel_regularizer=reg))(x)

    # --- Bidirectional GRU ---
    x = layers.Bidirectional(
        layers.GRU(32, return_sequences=True, dropout=0.3,
                   recurrent_dropout=0.2, kernel_regularizer=reg))(x)

    # --- Attention ---
    x = Attention()(x)

    # --- Classifier ---
    x = layers.Dense(64, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='SpeechCRNN')


# ======================== Custom Training Loop ========================
class SpecAugmentDataGen(keras.utils.Sequence):
    """Generator that applies SpecAugment on-the-fly each batch."""

    def __init__(self, X, y, sample_weights, batch_size=16, augment=True):
        self.X = X
        self.y = y
        self.sw = sample_weights
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        bi = self.indices[idx * self.batch_size:(idx+1) * self.batch_size]
        X_batch = self.X[bi].copy()
        y_batch = self.y[bi]
        sw_batch = self.sw[bi]

        if self.augment:
            X_batch = apply_spec_augment(X_batch)

        return X_batch, y_batch, sw_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ======================== Training ========================
def main():
    print("Loading data...")
    data = np.load(DATA_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    CLASSES = list(data['classes'])

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")

    # --- Class weights -> sample weights ---
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    sw_train = np.array([cw_dict[c] for c in y_train], dtype=np.float32)
    sw_val = np.array([cw_dict.get(c, 1.0) for c in y_val], dtype=np.float32)
    print(f"Class weights: {cw_dict}")

    # --- Convert to one-hot ---
    num_classes = len(CLASSES)
    y_train_oh = keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = keras.utils.to_categorical(y_val,   num_classes)
    y_test_oh  = keras.utils.to_categorical(y_test,  num_classes)

    # --- Generators ---
    train_gen = SpecAugmentDataGen(X_train, y_train_oh, sw_train, BATCH_SIZE, augment=True)
    val_gen   = SpecAugmentDataGen(X_val, y_val_oh, sw_val, BATCH_SIZE, augment=False)

    # --- Build & compile ---
    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    # --- Callbacks ---
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(
            MODEL_FILE, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    # --- Train ---
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("  SpecAugment + ClassWeights + LabelSmoothing(0.1)")
    print("  EarlyStopping(20) + ReduceLR(8)")
    print("=" * 60)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )

    # --- Load best model ---
    model = keras.models.load_model(
        MODEL_FILE, custom_objects={'Attention': Attention})

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=14)
    fig.colorbar(im)
    ticks = np.arange(len(CLASSES))
    ax.set_xticks(ticks)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(CLASSES, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150)
    print("Saved confusion_matrix.png")

    # --- Learning Curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'learning_curves.png'), dpi=150)
    print("Saved learning_curves.png")

    # --- Check for class dominance ---
    print("\n--- Class Balance Check ---")
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    true_counts = np.bincount(y_test, minlength=num_classes)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:30s}  true={true_counts[i]:3d}  pred={pred_counts[i]:3d}")

    max_pred = pred_counts.max()
    if max_pred > len(y_test) * 0.3:
        dominant = CLASSES[pred_counts.argmax()]
        print(f"\n  WARNING: {dominant} dominates predictions ({max_pred}/{len(y_test)})")
    else:
        print(f"\n  OK: No single class dominates predictions.")

    print(f"\nModel saved to: {MODEL_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
