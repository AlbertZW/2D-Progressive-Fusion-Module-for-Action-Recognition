import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils import data

class Dataset_CRNN(data.Dataset):
	def __init__(self, data_path, res_size, folders, labels, sample_num, transform=None, frame_format='frame_{:03d}.jpg', sampling_strategy='balanced'):
		self.data_path = data_path
		self.res_size = res_size
		self.labels = labels
		self.folders = folders
		self.transform = transform
		self.frame_format = frame_format
		self.sampling_strategy = sampling_strategy

		self.sampleNum = sample_num

	def __len__(self):
		return len(self.folders)

	# random sampling within segment (TSN method)
	def random_sampling(self, frame_num, segment_num=-1):
		index_list = []
		if segment_num == -1:
			segment_num = self.sampleNum
		frameInterval = frame_num // segment_num
		for s in range(segment_num):
			indexes = np.array(
				random.sample(range(0, frameInterval), self.sampleNum // segment_num)) + s * frameInterval * np.ones(
				self.sampleNum // segment_num)
			index_list += indexes.tolist()
		index_list.sort()
		return index_list


	def balanced_sampling(self, frame_num):
		index_list = []
		frameInterval = frame_num // self.sampleNum
		for s in range(self.sampleNum):
			index_list.append(s * frameInterval)
		return index_list


	def read_images(self, path, selected_folder, use_transform):
		X = []
		folder_path = os.path.join(path, selected_folder)
		frame_num = len(os.listdir(folder_path))

		if self.sampling_strategy == 'balanced':
			index_list = self.balanced_sampling(frame_num)
		elif self.sampling_strategy == 'random':
			index_list = self.random_sampling(frame_num)
		else:
			index_list = []

		for index in index_list:
			if index != -1:
				imagePath = os.path.join(folder_path, self.frame_format.format(int(index) + 1))
				image = [Image.open(imagePath).convert('RGB')]
			else:
				image = torch.zeros(3, self.res_size, self.res_size)
			X.extend(image)
		X = use_transform(X)
		return X

	def __getitem__(self, index):
		"Generates one sample of data"
		# Select sample
		folder = self.folders[index]

		# Load data
		X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
		y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

		# print(X.shape)
		return X, y
