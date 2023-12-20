import sys
import time

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset import PokemonTrainDataset, transform
from model import build_model

# class PredictDataset(Dataset):
#     def __init__(self, image_paths):
#         self.image_paths = image_paths
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#
#         image = Image.open(image_path).convert('RGB')
#
#         image_tensor = self.transform(image)
#
#         return image_path, image_tensor


device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

train_dataset = PokemonTrainDataset()
# predict_dataset = PredictDataset()
# predict_dataloader = DataLoader(predict_dataset, shuffle=True, batch_size=len(predict_dataset))

model = build_model(train_dataset.classes, for_train=False)
if torch.cuda.device_count() > 1 and device == "cuda":
    print("Launch parallel testing")
    model = torch.nn.DataParallel(model)

checkpoint = torch.load("pokeball.pth")
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded model, epochs: {checkpoint['epoch']}, accuracy {checkpoint['accuracy']}")
model.to(device)
model.eval()


def predict(img_tensors):
    print(f"Classifying {size(img_tensors)} images.")
    start = time.time()
    with torch.no_grad():
        y = model(img_tensors.to(device))
        probabilities = torch.nn.functional.softmax(y, 1)
        result = []
        for image_pro in probabilities:
            img_pros = []
            for i, p in enumerate(image_pro):
                img_pros.append((train_dataset.label_code_to_text(i), float(p)))
            img_pros.sort(key=lambda v: v[1], reverse=True)
            img_pros = img_pros[:3]
            # print(img_pros)
            result.append(img_pros)
        end = time.time()
        print(f"Classified images, costs {end - start} ms")
        return result


if __name__ == "__main__":
    X = []
    paths = sys.argv[1:]
    for img_path in paths:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image)
        X.append(image_tensor)

    X = torch.stack(X, 0).to(device)
    tasks = predict(X)
    for i, task in enumerate(tasks):
        if task[0][1] > 0.9:
            print(f"{paths[i]} is {task[0][0]} ( {task[0][1] * 100:.2f}% )")
        else:
            str = f"{paths[i]} maybe "
            for prob in task:
                str += f"{prob[0]} ({prob[1] * 100:.2f}%) "
                # print(f"{paths[i]} is ")
            print(str)
