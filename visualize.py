import os
import matplotlib.pyplot as plt
from PIL import Image


input_folder = "datasets/images"
mask_folder = "datasets/masks"
pred_folder = "outputs/predictions"
save_folder = "outputs/visualizations"
os.makedirs(save_folder, exist_ok=True)


image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]


for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    pred_path = os.path.join(pred_folder, filename)
    mask_path = os.path.join(mask_folder, filename)  

    
    input_img = Image.open(img_path).convert("RGB")
    pred_img = Image.open(pred_path).convert("L")
    mask_img = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else None

    # Plot
    fig, axs = plt.subplots(1, 3 if mask_img else 2, figsize=(12, 4))
    axs[0].imshow(input_img)
    axs[0].set_title("Input")
    axs[0].axis("off")

    if mask_img:
        axs[1].imshow(mask_img, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

    axs[-1].imshow(pred_img, cmap='gray')
    axs[-1].set_title("Prediction")
    axs[-1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_folder, f"viz_{filename}")
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    plt.close()
