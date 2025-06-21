from models.u2net import U2NET
import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np


device = torch.device('cpu')


model = U2NET().to(device)
model.load_state_dict(torch.load("outputs/checkpoints/u2net_epoch4.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


input_folder = "datasets/images"
output_folder = "outputs/predictions"
os.makedirs(output_folder, exist_ok=True)

def predict_all():
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(input_tensor)[0][0].cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                output_path = os.path.join(output_folder, filename)
                Image.fromarray(pred).save(output_path)
                print(f"Saved prediction: {output_path}")

predict_all()
