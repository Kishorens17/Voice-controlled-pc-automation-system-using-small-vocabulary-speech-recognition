"""Evaluate the best saved model."""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================== Config ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "processed_data.npz")
MODEL_FILE = os.path.join(BASE_DIR, "speech_crnn.keras")

# ======================== Attention Layer ========================
class Attention(keras.layers.Layer):
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

def main():
    print("Loading data...")
    data = np.load(DATA_FILE)
    X_test,  y_test  = data['X_test'],  data['y_test']
    CLASSES = list(data['classes'])

    print(f"Loading best model from {MODEL_FILE}...")
    model = keras.models.load_model(
        MODEL_FILE, custom_objects={'Attention': Attention}
    )

    y_test_oh  = keras.utils.to_categorical(y_test, len(CLASSES))
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    # Confusion matrix
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
    print("\nSaved confusion_matrix.png")

if __name__ == "__main__":
    main()
