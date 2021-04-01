
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

model_path = os.path.join('infer_model', 'resnet50_csv_36.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# load label to names mapping for visualization purposes
labels_to_names = {
    0: "No entry",
    1: "No parking / waiting",
    2: "No turning",
    3: "Max Speed",
    4: "Other prohibition signs",
    5: "Warning",
    6: "Mandatory"
}

# load image
image = read_image_bgr('data/za_traffic_2020/traffic_train/images/283.png')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)

plt.figure(figsize=(10, 5))
plt.axis('off')
plt.imshow(draw)
plt.tight_layout()
plt.show()