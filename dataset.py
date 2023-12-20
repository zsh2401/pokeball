import json
import math
import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cairosvg
import numpy as np
import cv2

# def get_transform():

# imageFilesDir = Path(r'./dataset/data/train')
# files = list(imageFilesDir.rglob('*.jpg'))
#
# # Since the std can't be calculated by simply finding it for each image and averaging like
# # the mean can be, to get the std we first calculate the overall mean in a first run then
# # run it again to get the std.
#
# mean = np.array([0., 0., 0.])
# stdTemp = np.array([0., 0., 0.])
# std = np.array([0., 0., 0.])
#
# numSamples = len(files)
#
# print("Calculating transform")
# for i in range(numSamples):
#     im = cv2.imread(str(files[i]))
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     im = im.astype(float) / 255.
#
#     for j in range(3):
#         mean[j] += np.mean(im[:, :, j])
#
# mean = (mean / numSamples)
#
# for i in range(numSamples):
#     im = cv2.imread(str(files[i]))
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     im = im.astype(float) / 255.
#     for j in range(3):
#         stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])
#
# std = np.sqrt(stdTemp / numSamples)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 图像标准化
                         std=[0.229, 0.224, 0.225])
])

print("Transform prepared")


class PokemonDataset(Dataset):
    def __init__(self, from_percent, to_percent):
        # 将图片进行初步的处理，使其能被卷积网络接收
        self.transform = transform

        self.data = []
        self.labels = {}
        self.__init_label_lib__("dataset")
        self.classes = len(self.labels)
        self.__prepare_data__("dataset", from_percent, to_percent)

        # print(f'数据集准备完毕，{len(self.data)}个样本，分布在{self.classes}个分类')

    def to_loader(self, batch_size=64):
        return DataLoader(self, batch_size=batch_size, shuffle=True)

    def __init_label_lib__(self, root_dir):
        with open('labels.json', 'r') as f:
            self.labels = json.load(f)

    def text_to_label_code(self, text):
        return self.labels[text]

    def label_code_to_text(self, code):
        for label in self.labels:
            if self.labels[label] == code:
                return label
        return None

    def __prepare_data__(self, root_dir, from_percent, to_percent):
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                files = os.listdir(label_path)
                start = math.floor(len(files) * from_percent)
                end_not_included = math.ceil(len(files) * to_percent)
                num = 0
                for i in range(start, end_not_included):
                    image_file = files[i]
                    if image_file.endswith(".svg"):
                        # print(f"Skipped {image_file}")
                        continue
                    image_path = os.path.join(label_path, image_file)
                    if os.path.isfile(image_path):
                        self.data.append((image_path, label))
                        num += 1
                # print(f"从 {label} 中采用了 {num}/{len(files)} ({(num / len(files) * 100):.2f}%) 个数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        if image_path.endswith(".svg"):
            # print(f"正在采用svg转png加载s{image_path}")
            out = BytesIO()
            cairosvg.svg2png(image_path, write_to=out)
            image = Image.open(out).convert("RGB")
            # print(f"采用svg转png加载了{image_path}")
        else:
            image = Image.open(image_path).convert('RGB')

        # 读取图片

        # 应用转换（如果有的话）
        # if self.transform:
        image_tensor = self.transform(image)

        return image_tensor, self.text_to_label_code(label), image_path, label


class PokemonTrainDataset(PokemonDataset):
    def __init__(self):
        super().__init__(0, 0.6)


class PokemonValidDataset(PokemonDataset):
    def __init__(self):
        super().__init__(0.6, 0.8)


class PokemonTestDataset(PokemonDataset):
    def __init__(self):
        super().__init__(0.8, 1)
