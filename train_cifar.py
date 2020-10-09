"""
Train on CIFAR-10/100 with (iterative) attentional feature fusion
============================
"""

from __future__ import division

import argparse
import time
import sys
import socket
import logging
import matplotlib
import numpy as np

import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
from gluoncv.data import transforms as gcv_transforms
from gluoncv.utils import makedirs, TrainingHistory, LRSequential, LRScheduler

from model import CIFARAFFResNet, CIFARAFFResNeXt
from utils import summary

matplotlib.use('Agg')
gcv.utils.check_version('0.6.0')


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='cifar10 or cifar100.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--askc-type', type=str, default='xxx',
                        help='ASKCFuse, ResGlobLocaforGlobLocaCha')
    parser.add_argument('--channel-times', type=int, default=4,
                        help='1, 2, 4.')
    parser.add_argument('--blocks', type=int, default=1,
                        help='1, 2, 3, 4, 5')
    parser.add_argument('--start-layer', type=int, default=1,
                        help='1, 2, 3, 4(no attention)')
    parser.add_argument('--summary', action='store_true',
                        help='print parameters')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--model', type=str, default='resnet',
                        help='resnet, resnext')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=96, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=640,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='learning rate. default is 0.2.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='300,450',
                        help='epochs at which learning rate decays. default is 300,450.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str, default='hybrid',
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--logging-dir', type=str, default='logs',
                        help='directory of training logs')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    parser.add_argument('--mixup', action='store_true',
                        help='use mixup. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=1,
                        help='beta distribution parameter for mixup sampling, default is 1.')
    parser.add_argument('--mixup-off-epoch', type=int, default=40,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--auto_aug', action='store_true',
                        help='use auto_aug. default is false.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--cosine', action='store_true',
                        help='use Cosine LR Decay in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='number of warmup epochs.')
    parser.add_argument('--deep-stem', action='store_true',
                        help='use deep-stem.')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    batch_size = opt.batch_size
    if opt.dataset == 'cifar10':
        classes = 10
    elif opt.dataset == 'cifar100':
        classes = 100
    else:
        raise ValueError('Unknown Dataset')

    if len(mx.test_utils.list_gpus()) == 0:
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in opt.gpus.split(',') if i.strip()]
        context = context if context else [mx.cpu()]
    print("context: ", context)
    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    num_workers = opt.num_workers

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        # lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
    num_batches = 50000 // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler('cosine', base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    optimizer = 'nag'
    if opt.cosine:
        optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
    else:
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

    layers = [opt.blocks] * 3
    channels = [x * opt.channel_times for x in [16, 16, 32, 64]]
    start_layer = opt.start_layer
    askc_type = opt.askc_type

    if channels[0] == 64:
        cardinality = 32
    elif channels[0] == 128:
        cardinality = 64
    bottleneck_width = 4

    print("model: ", opt.model)
    print("askc_type: ", opt.askc_type)
    print("layers: ", layers)
    print("channels: ", channels)
    print("start_layer: ", start_layer)
    print("classes: ", classes)
    print("deep_stem: ", opt.deep_stem)

    model_prefix = opt.dataset + '-' + askc_type
    model_suffix = '-c-' + str(opt.channel_times) + '-s-' + str(opt.start_layer)
    if opt.model == 'resnet':
        net = CIFARAFFResNet(askc_type=askc_type, start_layer=start_layer, layers=layers,
                             channels=channels, classes=classes, deep_stem=opt.deep_stem)
        model_name = model_prefix + '-resnet-' + str(sum(layers) * 2 + 2) + model_suffix
    elif opt.model == 'resnext':
        net = CIFARAFFResNeXt(askc_type=askc_type, start_layer=start_layer, layers=layers,
                              channels=channels, cardinality=cardinality,
                              bottleneck_width=bottleneck_width, classes=classes,
                              deep_stem=opt.deep_stem, use_se=False)
        model_name = model_prefix + '-resneXt-' + str(sum(layers) * 3 + 2) + '-' + \
                     str(cardinality) + 'x' + str(bottleneck_width) + model_suffix
    else:
        raise ValueError('Unknown opt.model')

    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context, ignore_extra=True)

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    plot_name = opt.save_plot_dir

    logging_handlers = [logging.StreamHandler()]
    if opt.logging_dir:
        logging_dir = opt.logging_dir
        makedirs(logging_dir)
        logging_handlers.append(logging.FileHandler('%s/train_%s.log' %
                                                    (logging_dir, model_name)))

    logging.basicConfig(level=logging.INFO, handlers=logging_handlers)
    logging.info(opt)

    transform_train = []
    if opt.auto_aug:
        print('Using AutoAugment')
        from autogluon.utils.augment import AugmentationBlock, autoaug_cifar10_policies
        transform_train.append(AugmentationBlock(autoaug_cifar10_policies()))

    transform_train.extend([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transform_train = transforms.Compose(transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    def label_transform(label, classes):
        ind = label.astype('int')
        res = nd.zeros((ind.shape[0], classes), ctx=label.context)
        res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1
        return res

    def mixup_transform(label, classes, lam=1, eta=0.0):
        if isinstance(label, nd.NDArray):
            label = [label]
        res = []
        for l in label:
            y1 = l.one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
            y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
            res.append(lam*y1 + (1-lam)*y2)
        return res

    def smooth(label, classes, eta=0.1):
        if isinstance(label, nd.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value=1 - eta + eta/classes, off_value = eta/classes)
            smoothed.append(res)
        return smoothed

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        if opt.summary:
            summary(net, mx.nd.zeros((1, 3, 32, 32), ctx=ctx[0]))
            sys.exit()

        if opt.dataset == 'cifar10':
            train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
                batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
            val_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif opt.dataset == 'cifar100':
            train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR100(train=True).transform_first(transform_train),
                batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
            val_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR100(train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            raise ValueError('Unknown Dataset')

        if opt.no_wd and opt.cosine:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

        if opt.label_smoothing or opt.mixup:
            sparse_label_loss = False
        else:
            sparse_label_loss = True

        metric = mx.metric.Accuracy()
        train_metric = mx.metric.RMSE()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)
        train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        lr_decay_count = 0

        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)

            if not opt.cosine:
                if epoch == lr_decay_epoch[lr_decay_count]:
                    trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                    lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data_1 = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label_1 = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                if opt.mixup:
                    lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                    if (epoch >= epochs - opt.mixup_off_epoch) or not opt.mixup:
                        lam = 1

                    data = [lam * X + (1 - lam) * X[::-1] for X in data_1]

                    if opt.label_smoothing:
                        eta = 0.1
                    else:
                        eta = 0.0
                    label = mixup_transform(label_1, classes, lam, eta)

                elif opt.label_smoothing:
                    hard_label = label_1
                    label = smooth(label_1, classes)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                if opt.mixup:
                    output_softmax = [nd.SoftmaxActivation(out) for out in output]
                    train_metric.update(label, output_softmax)
                else:
                    if opt.label_smoothing:
                        train_metric.update(hard_label, output)
                    else:
                        train_metric.update(label, output)

                name, acc = train_metric.get()
                iteration += 1

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(ctx, val_data)
            train_history.update([acc, 1 - val_acc])
            train_history.plot(save_path='%s/%s_history.png' % (plot_name, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-%s-best.params' %
                                    (save_dir, best_val_score, model_name))

            name, val_acc = test(ctx, val_data)
            logging.info('[Epoch %d] train=%f val=%f loss=%f lr: %f time: %f' %
                         (epoch, acc, val_acc, train_loss, trainer.learning_rate,
                          time.time() - tic))

        host_name = socket.gethostname()
        with open(opt.dataset + '_' + host_name + '_GPU_' + opt.gpus + '_best_Acc.log', 'a') as f:
            f.write('best Acc: {:.4f}\n'.format(best_val_score))
        print("best_val_score: ", best_val_score)

    if not opt.summary:
        if opt.mode == 'hybrid':
            net.hybridize()
    train(opt.num_epochs, context)


if __name__ == '__main__':
    main()
