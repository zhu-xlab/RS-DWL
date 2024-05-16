import torch


def feat_weight(feat1, feat2, wgt=torch.tensor[0.5,0.5]):
	return feat1*wgt[0] + 