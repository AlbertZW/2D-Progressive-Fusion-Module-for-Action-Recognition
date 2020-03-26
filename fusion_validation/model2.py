from torch import nn

from processing_data.transforms import *
from torch.nn.init import normal, constant
from collections import OrderedDict

from models import temporal_interaction

class TemporalFusionNet(nn.Module):
	def __init__(self, num_class, num_segments,
	             backbone_model='resnet50', modality='RGB',
	             new_length=None, dropout=0.8, crop_num=1):

		super(TemporalFusionNet, self).__init__()
		self.category_num = num_class
		self.num_segments = num_segments
		self.modality = modality
		self.dropout = dropout
		self.crop_num = crop_num

		if new_length is None:
			self.new_length = 1 * 3 if modality == "RGB" else 5 * 2
		else:
			self.new_length = new_length

		self._prepare_backbone(backbone_model)
		if self.modality == 'Flow':
			print("Converting the ImageNet model to a flow init model")
			self.backbone_model = self._construct_flow_model(self.backbone_model)
			print("Done. Flow model ready...")

		backbone_output_dim = getattr(self.backbone_model, self.backbone_model.last_layer_name).in_features

		modules = nn.Sequential(*list(self.backbone_model.children()))

		# Only for Resnet50
		self.ConvS = modules[:4]
		self.BottleNeck1 = self.backbone_model.layer1  # output 256

		self.BottleNeck2 = self.backbone_model.layer2
		# self.fusion_conv1_1 = nn.Sequential(
		# 	nn.Conv2d(2, 1, kernel_size=(3, 3), padding=1),
		# 	nn.ReLU(inplace=True)
		# )
		# self.fusion_conv1_2 = nn.Sequential(
		# 	nn.Conv2d(3, 1, kernel_size=(1, 1), padding=1),
		# 	nn.ReLU(inplace=True)
		# )
		# self._init_fuse_weight(self.fusion_conv1_1)
		# self._init_fuse_weight(self.fusion_conv1_2)

		self.BottleNeck3 = temporal_interaction.TIMLayer(self.backbone_model.layer3, n_div=8, conv_core=3)
		# self.fusion_conv2_1 = nn.Sequential(
		# 	nn.Conv2d(2, 1, kernel_size=(3, 3), padding=1),
		# 	nn.ReLU(inplace=True)
		# )
		self.fusion_conv2_2 = nn.Sequential(
			nn.Conv2d(3, 1, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True)
		)
		# self._init_fuse_weight(self.fusion_conv2_1)
		self._init_fuse_weight(self.fusion_conv2_2)

		self.BottleNeck4 = temporal_interaction.TIMLayer(self.backbone_model.layer4, n_div=6, conv_core=3)
		# self.fusion_conv3_1 = nn.Sequential(
		# 	nn.Conv2d(2, 2, kernel_size=(3, 3), padding=1)
		# )
		# self.fusion_conv3_2 = nn.Sequential(
		# 	nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
		# )
		# self.fusion_conv3_3 = nn.Sequential(
		# 	nn.Conv2d(4, 4, kernel_size=(3, 3), padding=1)
		# )

		self.fusion_conv4_1 = nn.Sequential(
			nn.Conv2d(3, 1, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True)
		)
		self.fusion_conv4_2 = nn.Sequential(
			nn.Conv2d(4, 1, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True)
		)
		self.fusion_conv4_3 = nn.Sequential(
			nn.Conv2d(5, 1, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True)
		)
		self.fusion_conv4_4 = nn.Sequential(
			nn.Conv2d(6, 1, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True)
		)
		# self._init_iden_weight(self.fusion_conv3_1)
		# self._init_iden_weight(self.fusion_conv3_2)
		# self._init_iden_weight(self.fusion_conv3_3)
		self._init_fuse_weight(self.fusion_conv4_1)
		self._init_fuse_weight(self.fusion_conv4_2)
		self._init_fuse_weight(self.fusion_conv4_3)
		self._init_fuse_weight(self.fusion_conv4_4)

		self.globalPooling = nn.AvgPool2d(7)

		# STUB 1024
		self.classification_layers = nn.Sequential(OrderedDict([
			('dropout', nn.Dropout(p=self.dropout)),
			('fc_classif', nn.Linear(backbone_output_dim, num_class)),
		]))

		# self.shallow_backbone_modules = []
		# self.deep_backbone_modules = [self.ResNet]

	def _init_fuse_weight(self, layer):
		for m in layer.modules():
			if isinstance(m, torch.nn.Conv2d):
				weight = torch.zeros((m.out_channels, m.in_channels) + m.kernel_size)
				weight[:, :, 1, 1] = 1 / m.in_channels
				m.weight.data = weight

	def _init_iden_weight(self, layer):
		for m in layer.modules():
			if isinstance(m, torch.nn.Conv2d):
				weight = torch.zeros((m.out_channels, m.in_channels) + m.kernel_size)
				for i in range(m.out_channels):
					for j in range(m.in_channels):
						if i == j:
							weight[i, j, 1, 1] = 1
				m.weight.data = weight


	def _prepare_backbone(self, backbone_model):
		if 'resnet' in backbone_model:
			self.backbone_model = getattr(torchvision.models, backbone_model)(True)
			self.backbone_model.last_layer_name = 'fc'
			self.input_size = 224
			self.input_mean = [0.485, 0.456, 0.406]
			self.input_std = [0.229, 0.224, 0.225]
			if self.modality == 'Flow':
				self.input_mean = [0.5]
				self.input_std = [np.mean(self.input_std)]

		elif backbone_model == 'BNInception':
			import tf_model_zoo
			self.backbone_model = getattr(tf_model_zoo, backbone_model)()
			self.backbone_model.last_layer_name = 'fc'
			self.input_size = 224
			self.input_mean = [104, 117, 128]
			self.input_std = [1]
			if self.modality == 'Flow':
				self.input_mean = [128]

		elif 'inception' in backbone_model:
			import tf_model_zoo
			self.backbone_model = getattr(tf_model_zoo, backbone_model)()
			self.backbone_model.last_layer_name = 'classif'
			self.input_size = 299
			self.input_mean = [0.5]
			self.input_std = [0.5]
		else:
			raise ValueError('Unknown backbone model: {}'.format(backbone_model))

	def get_optim_policies(self):
		first_conv_weight = []
		first_conv_bias = []
		normal_weight = []
		normal_bias = []
		bn = []

		conv_cnt = 0
		bn_cnt = 0
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
				ps = list(m.parameters())
				conv_cnt += 1
				if conv_cnt == 1:
					first_conv_weight.append(ps[0])
					if len(ps) == 2:
						first_conv_bias.append(ps[1])
				else:
					normal_weight.append(ps[0])
					if len(ps) == 2:
						normal_bias.append(ps[1])
			elif isinstance(m, torch.nn.Linear):
				ps = list(m.parameters())
				normal_weight.append(ps[0])
				if len(ps) == 2:
					normal_bias.append(ps[1])

			elif isinstance(m, torch.nn.BatchNorm1d):
				bn.extend(list(m.parameters()))
			elif isinstance(m, torch.nn.BatchNorm2d):
				bn_cnt += 1
			elif len(m._modules) == 0:
				if len(list(m.parameters())) > 0:
					raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

		return [
			{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
			 'name': "first_conv_weight"},
			{'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
			 'name': "first_conv_bias"},
			{'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
			 'name': "normal_weight"},
			{'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
			 'name': "normal_bias"},
			{'params': bn, 'lr_mult': 1, 'decay_mult': 0,
			 'name': "BN scale/shift"},
		]


	def forward(self, x):
		num_segments = self.num_segments
		# B * 8 * C * H * W
		x = x.view((-1, self.new_length) + x.size()[-2:])
		# (B * 8) * C * H * W

		x = self.ConvS(x)
		x = self.BottleNeck1(x)
		x = self.BottleNeck2(x)  # output ch 512
		# x = x.view((-1, num_segments, 512) + x.size()[-2:])
		# # B * 8 * C * H * W
		#
		# #####################################################################################
		# # refold and fuse the tensor (no stub)
		# x = x.permute(0, 2, 1, 3, 4).contiguous()
		# x = x.view((-1, num_segments) + x.size()[-2:])
		# # (B * C) * 8 * H * W
		# num_segments = num_segments - 2
		# for i in range(num_segments):
		# 	if i == 0:
		# 		x_fused_2 = self.fusion_conv1_2(x[:, i: i + 3, :, :])
		# 	else:
		# 		x_fused_2 = torch.cat((x_fused_2, self.fusion_conv1_2(x[:, i: i + 3, :, :])), dim=1)
		# x = x_fused_2
		# # (B * C) * 6 * H * W
		# #####################################################################################
		#
		# x = x.view((-1, 512, num_segments) + x.size()[-2:])
		# x = x.permute(0, 2, 1, 3, 4).contiguous()
		# x = x.view((-1, 512) + x.size()[-2:])
		# (B * 6) * C * H * W

		x = self.BottleNeck3(x)  # output ch 1024
		x = x.view((-1, num_segments, 1024) + x.size()[-2:])
		# B * 8 * C * H * W

		#####################################################################################
		# refold and fuse the tensor (no stub)
		x = x.permute(0, 2, 1, 3, 4).contiguous()
		x = x.view((-1, num_segments) + x.size()[-2:])
		# (B * C) * 8 * H * W
		num_segments = num_segments - 2
		for i in range(num_segments):
			if i == 0:
				# x_fused_1 = self.fusion_conv2_1(x[:, i: i + 2, :, :])
				x_fused_2 = self.fusion_conv2_2(x[:, i: i + 3, :, :])
			else:
				# x_fused_1 = torch.cat((x_fused_1, self.fusion_conv2_1(x[:, i: i + 2, :, :])), dim=1)
				x_fused_2 = torch.cat((x_fused_2, self.fusion_conv2_2(x[:, i: i + 3, :, :])), dim=1)
		# x = (x_fused_1 + 2 * x_fused_2) / 3
		x = x_fused_2
		# (B * C) * 6 * H * W
		#####################################################################################

		x = x.view((-1, 1024, num_segments) + x.size()[-2:])
		x = x.permute(0, 2, 1, 3, 4).contiguous()
		x = x.view((-1, 1024) + x.size()[-2:])
		# (B * 6) * C * H * W

		x = self.BottleNeck4(x)  # output ch 2048
		x = x.view((-1, num_segments, 2048) + x.size()[-2:])
		# B * 6 * C * H * W

		#####################################################################################
		# refold and fuse the tensor (no stub)
		x = x.permute(0, 2, 1, 3, 4).contiguous()
		x = x.view((-1, num_segments) + x.size()[-2:])
		# (B * C) * 6 * H * W
		num_segments = num_segments - 2
		for i in range(num_segments):
			if i == 0:
				x_fused_1 = self.fusion_conv4_1(x[:, i: i + 3, :, :])
				x_fused_2 = self.fusion_conv4_2(x[:, i: i + 4, :, :])
				x_fused_3 = self.fusion_conv4_3(x[:, i: i + 5, :, :])
				x_fused_4 = self.fusion_conv4_4(x[:, i: i + 6, :, :])
			else:
				if i < 2:
					x_fused_1 = torch.cat((x_fused_1, self.fusion_conv4_1(x[:, i: i + 3, :, :])), dim=1)
					x_fused_2 = torch.cat((x_fused_2, self.fusion_conv4_2(x[:, i: i + 4, :, :])), dim=1)
					x_fused_3 = torch.cat((x_fused_3, self.fusion_conv4_3(x[:, i: i + 5, :, :])), dim=1)
				elif i < 3:
					x_fused_1 = torch.cat((x_fused_1, self.fusion_conv4_1(x[:, i: i + 3, :, :])), dim=1)
					x_fused_2 = torch.cat((x_fused_2, self.fusion_conv4_2(x[:, i: i + 4, :, :])), dim=1)
				else:
					x_fused_1 = torch.cat((x_fused_1, self.fusion_conv4_1(x[:, i: i + 3, :, :])), dim=1)

		x_fused_1 = x_fused_1.mean(dim=1).unsqueeze(1)
		x_fused_2 = x_fused_2.mean(dim=1).unsqueeze(1)
		x_fused_3 = x_fused_3.mean(dim=1).unsqueeze(1)
		x = (x_fused_1 + x_fused_2 + 2 * x_fused_3 + 3 * x_fused_4) / 7
		# (B * C) * 1 * H * W
		#####################################################################################

		x = x.view((-1, 1, 2048) + x.size()[-2:])
		x = x.squeeze(1)
		x = self.globalPooling(x)
		x = x.squeeze(2)
		x = x.squeeze(2)

		x = self.classification_layers(x)

		return x

	def _construct_flow_model(self, backbone_model):
		# modify the convolution layers
		# Torch models are usually defined in a hierarchical way.
		# nn.modules.children() return all sub modules in a DFS manner
		modules = list(self.backbone_model.modules())
		first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
		conv_layer = modules[first_conv_idx]
		container = modules[first_conv_idx - 1]

		# modify parameters, assume the first blob contains the convolution kernels
		params = [x.clone() for x in conv_layer.parameters()]
		kernel_size = params[0].size()
		new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
		new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

		new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
		                     conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
		                     bias=True if len(params) == 2 else False)
		new_conv.weight.data = new_kernels
		if len(params) == 2:
			new_conv.bias.data = params[1].data  # add bias if neccessary
		layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

		# replace the first convlution layer
		setattr(container, layer_name, new_conv)
		return backbone_model

	@property
	def crop_size(self):
		return self.input_size

	@property
	def scale_size(self):
		return self.input_size * 256 // 224

	def get_augmentation(self):
		if self.modality == 'RGB':
			return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
			                                       GroupRandomHorizontalFlip(is_flow=False)])
		elif self.modality == 'Flow':
			return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
			                                       GroupRandomHorizontalFlip(is_flow=True)])
		elif self.modality == 'RGBDiff':
			return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
			                                       GroupRandomHorizontalFlip(is_flow=False)])
