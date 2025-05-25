import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import Button, HBox, VBox, Output
from IPython.display import display, clear_output


# 绘图相关代码，不用动
def draw_boxes(img, predictions, ax, title=""):
    img = img.detach().cpu().numpy()  # Convert tensor to numpy array
    img = np.transpose(img, (1, 2, 0))  # Change the order of dimensions
    img = np.clip(img, 0, 1) * 255  # Convert image back to pixel range [0, 255]
    img = img.astype(np.uint8)  # Convert to uint8 for cv2
    
    print(f'{title}: pred result:')
    
    for pred in predictions:
        box = pred['box']
        score = pred['score']
        label = pred['label']

        # Draw bounding box
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # Modify text parameters here:
        text = f'{label} ({score:.2f})'
        print(text)
        font_scale = 0.8  # 原值是0.5，调大数值可增大字体（例如改为0.8或1.0）
        thickness = 2     # 字体边框粗细，可与font_scale配合调整

        # 使用更清晰的字体（可选）
        font = cv2.FONT_HERSHEY_DUPLEX  # 原默认是cv2.FONT_HERSHEY_SIMPLEX

        img = cv2.putText(
            img,
            text,
            (int(box[0]), int(box[1]) - 10),
            font,           # 使用新字体
            font_scale,     # 新字体大小
            (255, 0, 0),   # 文字颜色
            thickness       # 字体粗细
        )

    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')    
    
# Function to visualize a pair of images
def visualize_single_image(model, index, dataset):
    clear_output() 
    img = dataset[index]["image"]
    preds = model(img.unsqueeze(0))
    results = [
        {'box': pred[:4], 'score': pred[4], 'label': model.names[int(pred[5])]}
        for pred in preds[0].boxes.data
    ]
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 1, figsize=(30, 14))
    draw_boxes(img.squeeze(), results, ax, title="Image Predictions")
    plt.show()

# Function to visualize a pair of images
def visualize_image_pair(model, index, origin_dataset, attacked_dataset):
    clear_output() 
    # Get the original and adversarial images
    img = origin_dataset[index]["image"]
    perturbed_img = attacked_dataset[index]["image"]

    # Get predictions for both images
    original_preds = model(img.unsqueeze(0))
    adversarial_preds = model(perturbed_img.unsqueeze(0))

    # Extract useful prediction data (bounding boxes, scores, and labels)
    original_results = [
        {'box': pred[:4], 'score': pred[4], 'label': model.names[int(pred[5])]}
        for pred in original_preds[0].boxes.data
    ]
    adversarial_results = [
        {'box': pred[:4], 'score': pred[4], 'label': model.names[int(pred[5])]}
        for pred in adversarial_preds[0].boxes.data
    ]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(30, 14))

    # Draw the original image with predictions
    draw_boxes(img.squeeze(), original_results, axs[0], title="Original Image Predictions")

    # Draw the adversarial image with predictions
    draw_boxes(perturbed_img.squeeze(), adversarial_results, axs[1], title="Adversarial Image Predictions")

    # Show the plot
    plt.show()
    
    
def visualize_attack_result(model, origin_dataset, attacked_dataset):
    dataset_size = len(origin_dataset)
    current_index = 5

    prev_button = Button(description="Previous")
    next_button = Button(description="Next")
    index_label = Button(description=f"Current Index: {current_index}", disabled=True)
    output = Output()

    def update_display(index):
        nonlocal current_index
        current_index = index
        index_label.description = f"Current Index: {current_index}"
        with output:
            clear_output()
            visualize_image_pair(model, current_index, origin_dataset, attacked_dataset)

    def on_prev_click(b):
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_display(current_index)

    def on_next_click(b):
        nonlocal current_index
        if current_index < dataset_size - 1:
            current_index += 1
            update_display(current_index)

    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)

    update_display(0)

    controls = HBox([prev_button, next_button, index_label])
    display(VBox([controls, output]))