import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TFNDataSet
from models.TemporalFusionNet import TemporalFusionNet
from processing_data.transforms import *


def no_rnn(args):
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

    elif args.dataset == 'somethingV2':
        data_path = '/data/Disk_C/something/20bn-something-something-v2_images'
        if args.modality == 'Flow':
            pass
        list_folder = './data_ckpt/somethingV2'
        save_model_path = "./data_ckpt/somethingV2_ckpt/"
        frame_format = '{:04d}.jpg'
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

    processing_model = TemporalFusionNet(num_class, args.num_segments,
                                         backbone_model=args.arch,
                                         modality=args.modality,
                                         dropout=args.dropout
                                         )

    crop_size = processing_model.crop_size
    scale_size = processing_model.scale_size
    input_mean = processing_model.input_mean
    input_std = processing_model.input_std
    policies = processing_model.get_optim_policies()
    train_augmentation = processing_model.get_augmentation()

    processing_model = torch.nn.DataParallel(processing_model, device_ids=args.gpus).cuda()

    best_prec1 = 0

    snapshot_pref = '_'.join((args.dataset, args.arch))

    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    if args.resume:
        filename = '_'.join((snapshot_pref, args.modality.lower(), 'checkpoint.pth.tar'))
        resume_path = os.path.join(save_model_path, filename)
        if os.path.isfile(resume_path):
            print(("=> loading checkpoint '{}'".format(resume_path)))
            checkpoint = torch.load(resume_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            processing_model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch'])))

            epoch_train_losses = np.load(os.path.join(save_model_path, 'training_losses.npy')).tolist()
            epoch_train_scores = np.load(os.path.join(save_model_path, 'training_scores.npy')).tolist()
            epoch_test_losses = np.load(os.path.join(save_model_path, 'test_loss.npy')).tolist()
            epoch_test_scores = np.load(os.path.join(save_model_path, 'test_score.npy')).tolist()
        else:
            print(("=> no checkpoint found at '{}'".format(resume_path)))

    if args.applyPretrainedModel:
        filename = '_'.join(('pretrained', args.modality.lower(), 'checkpoint.pth.tar'))
        resume_path = os.path.join(save_model_path, filename)
        if os.path.isfile(resume_path):
            print(("=> loading checkpoint '{}'".format(resume_path)))
            pretrained_dict = torch.load(resume_path)['state_dict']
            model_dict = processing_model.state_dict()

            regarded_dict = {}
            for k, v in pretrained_dict.items():
                for curr_k in model_dict.keys():
                    if k == curr_k and v.size() == model_dict[curr_k].size():
                        regarded_dict[k] = v

            model_dict.update(regarded_dict)
            processing_model.load_state_dict(model_dict)
            print(("=> loaded checkpoint '{}'".format(filename)))
        else:
            print(("=> no checkpoint found at '{}'".format(resume_path)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TFNDataSet(data_path, os.path.join(list_folder, args.train_list), num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=frame_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception', isolated=True),
                       ToTorchFormatTensor(div=args.arch != 'BNInception', isolated=True),
                       normalize,
                   ])),
        batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TFNDataSet(data_path, os.path.join(list_folder, args.val_list), num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=frame_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
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

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        filename = '_'.join((snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        eval_path = os.path.join(save_model_path, filename)
        if os.path.isfile(eval_path):
            print(("=> loading checkpoint '{}'".format(eval_path)))
            checkpoint = torch.load(eval_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            processing_model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))

        else:
            print(("=> no checkpoint found at '{}'".format(eval_path)))
            return

        validate(val_loader, processing_model, criterion, 0, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_steps, args.weight_decay)

        # train for one epoch
        train_accu, train_loss = train(train_loader, processing_model, criterion, optimizer, epoch,
                                       args.clip_gradient, args.print_freq)

        epoch_train_losses.append(train_loss)
        epoch_train_scores.append(train_accu)

        tmp_training_loss_path = os.path.join(save_model_path, 'tmp_training_losses.npy')
        tmp_training_score_path = os.path.join(save_model_path, 'tmp_training_scores.npy')
        np.save(tmp_training_loss_path, np.array(epoch_train_losses))
        np.save(tmp_training_score_path, np.array(epoch_train_scores))

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, val_loss = validate(val_loader, processing_model, criterion, (epoch + 1) * len(train_loader),
                                       args.print_freq)

            epoch_test_losses.append(val_loss)
            epoch_test_scores.append(prec1)
            np.save(os.path.join(save_model_path, 'test_loss.npy'), np.array(epoch_test_losses))
            np.save(os.path.join(save_model_path, 'test_score.npy'), np.array(epoch_test_scores))

            training_loss_path = os.path.join(save_model_path, 'training_losses.npy')
            training_score_path = os.path.join(save_model_path, 'training_scores.npy')
            shutil.copyfile(tmp_training_loss_path, training_loss_path)
            shutil.copyfile(tmp_training_score_path, training_score_path)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = prec1
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': processing_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, snapshot_pref, args.modality, save_model_path, epoch)


def train(train_loader, model, criterion, optimizer, epoch, clip_grad, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if clip_grad is not None:
            total_norm = clip_grad_norm(model.parameters(), clip_grad)
            # if total_norm > args.clip_gradient:
            #     print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, iter, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, snapshot_pref, modality, save_model_path, epoch=0, filename='checkpoint.pth.tar'):
    filename = '_'.join((snapshot_pref, modality.lower(), filename))
    file_path = os.path.join(save_model_path, filename)
    torch.save(state, file_path)
    if is_best:
        best_name = '_'.join((snapshot_pref, modality.lower(), 'model_best.pth.tar'))
        best_path = os.path.join(save_model_path, best_name)
        shutil.copyfile(file_path, best_path)

    # epoch count from 1 but actually starts from 0
    if 24 <= epoch <= 44:
        epochinfo = 'epoch' + str(epoch)
        backup_filename = '_'.join((snapshot_pref, modality.lower(), epochinfo, filename))
        backup_path = os.path.join(save_model_path, backup_filename)
        shutil.copyfile(file_path, backup_path)


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


def adjust_learning_rate(optimizer, epoch, arg_lr, lr_steps, arg_weight_decay):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = arg_lr * decay
    decay = arg_weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


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

