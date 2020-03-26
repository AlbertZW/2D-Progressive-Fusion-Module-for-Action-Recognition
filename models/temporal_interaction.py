import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalInteraction(nn.Module):
	def __init__(self, res_block, n_div=8, conv_core=3):
		super(TemporalInteraction, self).__init__()
		self.res_block = res_block
		self.n_div = n_div

		self.interaction_conv = nn.Conv2d(conv_core, 1, kernel_size=(1, 1))

	def forward(self, x):
		x = self.res_block(x)
		#
		nt, c, h, w = x.size()
		n_batch = nt // self.n_div
		x = x.view(n_batch, self.n_div, c, h, w)
		x = x.permute(0, 2, 1, 3, 4).contiguous()
		x = x.view(n_batch * c, self.n_div, h, w)

		zero_padding = torch.zeros_like(x[:, 0, :, :].unsqueeze(1))
		x = torch.cat((zero_padding, x), dim=1)
		x = torch.cat((x, zero_padding), dim=1)
		for i in range(self.n_div):
			if i == 0:
				x_fused = self.interaction_conv(x[:, i: i + 3, :, :])
			else:
				x_fused = torch.cat((x_fused, self.interaction_conv(x[:, i: i + 3, :, :])), dim=1)
		x = x_fused

		x = x.view((n_batch, -1, self.n_div) + x.size()[-2:])
		x = x.permute(0, 2, 1, 3, 4).contiguous()
		x = x.view((nt, -1) + x.size()[-2:])

		return x


def TIMLayer(resnet_layer, n_div=8, conv_core=3):
	blocks = list(resnet_layer.children())
	for i, b in enumerate(blocks):
		if i != len(blocks) - 1:
			blocks[i] = TemporalInteraction(b, n_div, conv_core)
	return nn.Sequential(*blocks)
