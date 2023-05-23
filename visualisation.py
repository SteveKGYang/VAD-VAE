import torch
import pickle
a = torch.load("./latent_save_dir/DailyDialog/content.pt")
print(a.shape)
file = open("./latent_save_dir/DailyDialog/labels.pt", "rb")
b = pickle.load(file)
print(len(b))