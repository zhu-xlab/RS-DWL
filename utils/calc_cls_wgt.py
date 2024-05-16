import torch
import torch.nn.functional as F


def freq_cls_wgt(loader):
    num_0, num_1 = 0, 0
    for step, (_, _, label, _) in enumerate(loader):
        num_0 += (label==0).float().sum()
        num_1 += (label==1).float().sum()

    num = num_0 + num_1
    freq_0 = num_1 / num
    freq_1 = num_0 / num
    weight = torch.tensor([freq_0, freq_1])

    return weight

