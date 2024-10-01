import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


unet_plus_plus.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coef])
attention_unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coef])


def load_and_preprocess_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = apply_clahe(img)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)  # Convert grayscale to RGB
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def visualize_segmentation(image_path, model):
    image = load_and_preprocess_dicom(image_path)
    prediction = model.predict(image)[0]
    prediction = (prediction > 0.5).astype(np.uint8)

    # Load original image for display
    dicom = pydicom.dcmread(image_path)
    original = dicom.pixel_array
    original = cv2.resize(original, (256, 256))

    # Apply CLAHE for better visibility
    original = apply_clahe(original)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Segmentation')
    plt.imshow(original, cmap='gray')
    plt.imshow(prediction[:, :, 0], cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

