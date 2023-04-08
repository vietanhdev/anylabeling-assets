import numpy as np
import cv2
import matplotlib.pyplot as plt
import onnxruntime
from copy import deepcopy

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def apply_coords(coords: np.ndarray, original_size, target_length) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length
    )
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

image = cv2.imread('docs/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

encoder_path = "./vit_b-encoder-quant.onnx"
decoder_path = "./vit_b-decoder-quant.onnx"

encoder_session = onnxruntime.InferenceSession(encoder_path)
decoder_session = onnxruntime.InferenceSession(decoder_path)

# Rewrite code without torch
image_size = 1024

# Resize longest side
original_size = image.shape[:2]
h, w = image.shape[:2]
if h > w:
    new_h, new_w = image_size, int(w * image_size / h)
else:
    new_h, new_w = int(h * image_size / w), image_size
input_image = cv2.resize(image, (new_w, new_h))

# Normalize
pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, -1)
pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, -1)
x = (input_image - pixel_mean) / pixel_std

# Padding to square
h, w = x.shape[:2]
padh = image_size - h
padw = image_size - w
x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant')
x = x.astype(np.float32)

# Transpose
x = x.transpose(2, 0, 1)[None, :, :, :]

encoder_inputs = {
    "x": x,
}

output = encoder_session.run(None, encoder_inputs)
image_embedding = output[0]


# ===== Points

input_point = np.array([[500, 375]])
input_label = np.array([1])

# Add a batch index, concatenate a padding point, and transform.
onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
onnx_coord = apply_coords(onnx_coord, original_size, image_size).astype(np.float32)

# Create an empty mask input and an indicator for no mask.
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)


decoder_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}
masks, _, low_res_logits = decoder_session.run(None, decoder_inputs)
masks = masks > 0.0


plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()


# ===== Box and points

input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])

onnx_box_coords = input_box.reshape(2, 2)
onnx_box_labels = np.array([2,3])

onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)
onnx_coord = apply_coords(onnx_coord, original_size, image_size).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

masks, _, _ = decoder_session.run(None, ort_inputs)
masks = masks > 0.0


plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()