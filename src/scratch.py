import os
import os.path as osp

from torch.utils import data
import numpy as np

from loader import Dataset, load_data

# dataset = Dataset("data/train", "acl", True)

# loader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

# for batch in loader:
#   for thing in batch:
#     try:
#       print(thing.shape)
#     except Exception as e:
#       print(e)
#   break

data_dir = 'data/train/axial'
data_files = os.listdir(data_dir)
channels = []
for i, fname in enumerate(data_files):
  data_array = np.load(osp.join(data_dir, fname))
  channels.append(data_array.shape[0])
channels = np.array(channels)

print("min:", channels.min(), "max:", channels.max(), "mean:", channels.mean())
print(channels)
