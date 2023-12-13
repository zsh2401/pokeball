import torch
from torch.utils.data import DataLoader

from dataset import PokemonTrainDataset
from model import Pokeball

DEVICE = torch.device("mps")
train_dataset = PokemonTrainDataset()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
model = Pokeball(num_classes=train_dataset.classes, device=DEVICE)
model.load_state_dict(torch.load("./serena.model"))


def predict(img_tensors):
    Y = model(img_tensors)

# predict()
