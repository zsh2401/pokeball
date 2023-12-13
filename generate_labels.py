import json
import os

from sklearn.preprocessing import LabelEncoder

labels = {}
for label in os.listdir("./dataset"):
    label_path = os.path.join("./dataset", label)
    if os.path.isdir(label_path):
        labels[label] = True

label_encoder = LabelEncoder()

# 将标签列表转换为整数码列表
list_labels = list(labels)
integer_labels = label_encoder.fit_transform(list_labels)
for i in range(0, len(list_labels)):
    labels[list_labels[i]] = int(integer_labels[i])

with open('labels.json', 'w') as f:
    json.dump(labels, f)
