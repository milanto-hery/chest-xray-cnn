import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def grad_cam(model, img_path, layer_name):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224,224))
    input_tensor = np.expand_dims(img_resized/255.0, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        loss = tf.reduce_max(predictions)

    grads = tape.gradient(loss, conv_outputs)[0]
    heatmap = np.mean(grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    output = heatmap * 0.4 + img
    plt.imshow(output.astype("uint8"))
    plt.axis("off")
    plt.show()
