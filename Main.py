import os

class Args:
	def __init__(self, dataset, segment_num, dropout, backbone_arch, gpus,
	             epochs, train_batch_size, val_batch_size, lr_steps, resume=False, evaluate=False,
	             applyPretrainedModel=False, start_epoch=0, modality='RGB', flow_prefix='flow_'):
		self.dataset = dataset
		self.num_segments = segment_num
		self.dropout = dropout
		self.arch = backbone_arch
		self.gpus = gpus
		self.epochs = epochs
		self.train_batch_size = train_batch_size
		self.val_batch_size = val_batch_size
		self.lr_steps = lr_steps

		self.start_epoch = start_epoch
		self.resume = resume
		self.evaluate = evaluate
		self.applyPretrainedModel = applyPretrainedModel
		self.modality = modality
		self.flow_prefix = flow_prefix


		self.lr = 0.001
		self.clip_gradient = 20
		self.print_freq = 20
		self.eval_freq = 5
		self.workers = 4
		self.train_list = 'trainlist.txt'
		self.val_list = 'testlist.txt'
		self.loss_type = 'nll'
		self.momentum = 0.9
		self.weight_decay = 5e-4


if __name__ == "__main__":

	args = Args(
		dataset='UCF101',
		segment_num=8,
		dropout=0.8,
		backbone_arch='resnet50',
		# backbone_arch='BNInception',
		gpus=[0],
		epochs=70,
		train_batch_size=2,
		val_batch_size=64,
		lr_steps=[30, 50],
		# resume=True,
		# evaluate=True,

		applyPretrainedModel=True,
	)

	fusion_val = False

	if fusion_val:
		from fusion_validation.fusion_validation import fusion_validation
		fusion_validation(args)
	else:
		from no_rnn import no_rnn
		no_rnn(args)

