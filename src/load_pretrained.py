from model import TripleMRNet
import torch

model = TripleMRNet(backbone="resnet18")

save_path = "exp2/val0_train0_epoch0.pth"
torch.save(model.state_dict(), save_path)