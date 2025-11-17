import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)


# --- 1. U-Net Model Definition ---

def unet_model(input_size=(128, 128, 1), num_classes=1):
    """
    Constructs the U-Net model architecture.

    Args:
        input_size (tuple): The dimensions of the input image (H, W, C).
        num_classes (int): The number of classes to segment (1 for binary).
    """
    inputs = Input(input_size)

    # --------------------------------
    # ENCODING PATH (Downsampling)
    # --------------------------------

    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # --------------------------------
    # BOTTLENECK
    # --------------------------------

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    # --------------------------------
    # DECODING PATH (Upsampling & Skip Connections)
    # --------------------------------

    # Block 5 (Merging with conv3)
    up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)  # Skip connection
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # Block 6 (Merging with conv2)
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)  # Skip connection
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # Block 7 (Merging with conv1)
    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)  # Skip connection
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # --------------------------------
    # OUTPUT
    # --------------------------------

    # Final layer uses Sigmoid for binary segmentation (output pixel probability)
    conv8 = Conv2D(num_classes, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    # Using Adam optimizer and binary cross-entropy loss for binary segmentation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# --- 2. Synthetic Data Generator ---

def generate_synthetic_data(num_samples, img_size):
    """Generates simple circular images and corresponding binary masks."""
    X = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    Y = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    center = img_size // 2

    for i in range(num_samples):
        # Create a circle mask
        radius = np.random.randint(5, center - 5)
        cy = np.random.randint(center - 20, center + 20)
        cx = np.random.randint(center - 20, center + 20)

        # Draw mask
        for y in range(img_size):
            for x in range(img_size):
                if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                    Y[i, y, x, 0] = 1.0  # White pixel in mask
                    X[i, y, x, 0] = np.random.uniform(0.7, 1.0)  # Bright pixel in input
                else:
                    X[i, y, x, 0] = np.random.uniform(0.0, 0.3)  # Dark pixel in input

    return X, Y


# --- 3. Execution ---

IMG_SIZE = 128
NUM_SAMPLES = 50
EPOCHS = 5

# Generate data
X, Y = generate_synthetic_data(NUM_SAMPLES, IMG_SIZE)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Build model
unet = unet_model(input_size=(IMG_SIZE, IMG_SIZE, 1), num_classes=1)

print("\n--- U-Net Model Summary ---")
unet.summary()

# Train model
print("\n--- Starting U-Net Training ---")
history = unet.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=8,
    validation_data=(X_test, Y_test),
    verbose=1
)

# --- 4. Verification (Prediction and Visualization) ---

# Predict on test data
predictions = unet.predict(X_test)

# Choose a sample to visualize
sample_index = 0
predicted_mask = (predictions[sample_index, ..., 0] > 0.5).astype(np.uint8)  # Binarize output

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Input Image
axes[0].imshow(X_test[sample_index, ..., 0], cmap='gray')
axes[0].set_title('Original Input Image')
axes[0].axis('off')

# Ground Truth Mask
axes[1].imshow(Y_test[sample_index, ..., 0], cmap='gray')
axes[1].set_title('Ground Truth Mask')
axes[1].axis('off')

# Predicted Mask
axes[2].imshow(predicted_mask, cmap='gray')
axes[2].set_title('Predicted Mask (Threshold 0.5)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("\n--- Verification Complete ---")
print(f"Final Test Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final Test Accuracy: {history.history['val_accuracy'][-1]:.4f}")