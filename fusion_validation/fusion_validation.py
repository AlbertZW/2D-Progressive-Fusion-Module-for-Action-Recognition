import os
import time
import torch
import torchvision
from dataset import TFNDataSet
from processing_data.transforms import *
from fusion_validation import model1
from fusion_validation import model2


def fusion_validation(args):
	if args.dataset == 'Kinetics400':
		data_path = "/data/Disk_C/Kinetics400_Resources/"
		if args.modality == 'Flow':
			pass
		list_folder = './data_ckpt/kinetics400'
		save_model_path = "./data_ckpt/kinetics400_ckpt/"
		frame_format = 'frame_{:04d}.jpg'
		num_class = 400  # number of target category

	elif args.dataset == 'somethingV1':
		data_path = '/data/Disk_C/something/20bn-something-something-v1'
		if args.modality == 'Flow':
			pass
		list_folder = './data_ckpt/somethingV1'
		save_model_path = "./data_ckpt/somethingV1_ckpt/"
		frame_format = '{:05d}.jpg'
		num_class = 174

	elif args.dataset == 'UCF101':
		# data_path = '/data/Disk_C/UCF101_Resources/UCF-101_IMAGES/'
		# data_path = '/media/albert/DATA1/_DataSources/UCF-101_IMAGES/'
		data_path = 'D:\\_Datasets\\UCF-101_IMAGES'
		if args.modality == 'Flow':
			pass
		list_folder = './data_ckpt/UCF101'
		save_model_path = "./data_ckpt/UCF101_ckpt/"
		frame_format = 'frame_{:03d}.jpg'
		num_class = 101

	else:
		raise ValueError('Unknown dataset ' + args.dataset)

	first_model = model1.TemporalFusionNet(num_class, args.num_segments,
	                                     backbone_model=args.arch,
	                                     modality=args.modality,
	                                     dropout=args.dropout
	                                     )


	crop_size = first_model.crop_size
	scale_size = first_model.scale_size
	input_mean = first_model.input_mean
	input_std = first_model.input_std

	normalize = GroupNormalize(input_mean, input_std)

	first_model = torch.nn.DataParallel(first_model, device_ids=args.gpus).cuda()

	second_model = model2.TemporalFusionNet(num_class, args.num_segments,
	                                     backbone_model=args.arch,
	                                     modality=args.modality,
	                                     dropout=args.dropout
	                                     )
	second_model = torch.nn.DataParallel(second_model, device_ids=args.gpus).cuda()

	best_prec1 = 0
	snapshot_pref = '_'.join((args.dataset, args.arch))

	if args.modality == 'RGB':
		data_length = 1
	elif args.modality in ['Flow', 'RGBDiff']:
		data_length = 5

	val_loader = torch.utils.data.DataLoader(
		TFNDataSet(data_path, os.path.join(list_folder, args.val_list), num_segments=args.num_segments,
		           new_length=data_length,
		           modality=args.modality,
		           image_tmpl=frame_format if args.modality in ["RGB",
		                                                        "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
		           random_shift=False,
		           transform=torchvision.transforms.Compose([
			           GroupScale(int(scale_size)),
			           GroupCenterCrop(crop_size),
			           Stack(roll=args.arch == 'BNInception', isolated=True),
			           ToTorchFormatTensor(div=args.arch != 'BNInception', isolated=True),
			           normalize,
		           ])),
		batch_size=args.val_batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)



	filename1 = '_'.join((snapshot_pref, 'first_model.pth.tar'))
	eval_path1 = os.path.join(save_model_path, filename1)
	filename2 = '_'.join((snapshot_pref, 'second_model.pth.tar'))
	eval_path2 = os.path.join(save_model_path, filename2)
	if os.path.isfile(eval_path1) and os.path.isfile(eval_path2):

		print(("=> loading models '{}' and '{}'".format(eval_path1, eval_path2)))
		checkpoint1 = torch.load(eval_path1)
		checkpoint2 = torch.load(eval_path2)
		first_model.load_state_dict(checkpoint1['state_dict'])
		second_model.load_state_dict(checkpoint2['state_dict'])

		print("=> Models loaded. Evaluation started.")

	else:
		print('Incomplete trained model.')
		return

	batch_time = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	first_model.eval()
	second_model.eval()

	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		with torch.no_grad():
			output1 = first_model(input_var)
			output2 = second_model(input_var)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(output1.data + output2.data, target, topk=(1, 5))
			top1.update(prec1.item(), input.size(0))
			top5.update(prec5.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 20 == 0:
				print(('Test: [{0}/{1}]\t'
				       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					i, len(val_loader), batch_time=batch_time, top1=top1, top5=top5)))

		print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
		       .format(top1=top1, top5=top5)))


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res
