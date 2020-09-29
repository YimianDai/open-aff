"""
Train on CIFAR-10/100 with Mixup
============================
"""

from __future__ import division

import matplotlib
import argparse, time, logging, random, math
import numpy as np
import mxnet as mx
import socket
import sys

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
from gluoncv.data import transforms as gcv_transforms
from gluoncv.utils import makedirs, TrainingHistory

from model import ResNet110V2ASKC, CIFARResNextASKC
from utils import summary

matplotlib.use('Agg')
gcv.utils.check_version('0.6.0')


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='cifar10 or cifar100.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--askc-type', type=str, default='xxx',
                        help='ASKCFuse, ResLocaLocaCha, ResGlobGlobCha')
    parser.add_argument('--channel-times', type=int, default=1,
                        help='1, 2, 4.')
    parser.add_argument('--blocks', type=int, default=1,
                        help='1, 2, 3, 4, 5')
    parser.add_argument('--start-layer', type=int, default=1,
                        help='1, 2, 3, 4(no attention)')
    parser.add_argument('--summary', action='store_true',
                        help='print parameters')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--model', type=str, default='resnet',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str,
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
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

    # layers = [18, 18, 18]

    if opt.model == 'resnet':
        layers = [opt.blocks] * 3
        channels = [x * opt.channel_times for x in [16, 16, 32, 64]]
        start_layer = opt.start_layer

        net = ResNet110V2ASKC(in_askc_type=opt.askc_type, layers=layers, channels=channels, classes=classes, start_layer=start_layer)
        model_name = 'resnet_' + opt.askc_type + '_blocks_' + str(opt.blocks) + '_channel_times_' + \
                     str(opt.channel_times) + '_start_layer_' + str(start_layer) + '_mixup'
        print("opt.askc_type: ", opt.askc_type)
        print("layers: ", layers)
        print("channels: ", channels)
        print("classes: ", classes)
        print("start_layer: ", start_layer)
    elif opt.model == 'resnext29_32x4d':
        num_layers = 29
        layer = (num_layers - 2) // 9
        layers = [layer] * 3
        cardinality = 32
        bottleneck_width = 4
        net = CIFARResNextASKC(layers, cardinality, bottleneck_width, classes, use_se=False)
        model_name = 'resnext29_32x4d_askc_mixup'
    elif opt.model == 'resnext38_32x4d':
        num_layers = 38
        layer = (num_layers - 2) // 9
        layers = [layer] * 3
        cardinality = 32
        bottleneck_width = 4
        net = CIFARResNextASKC(layers, cardinality, bottleneck_width, classes, use_se=False)
        model_name = 'resnext47_32x4d_iaff_mixup'
    elif opt.model == 'resnext47_32x4d':
        num_layers = 47
        layer = (num_layers - 2) // 9
        layers = [layer] * 3
        cardinality = 32
        bottleneck_width = 4
        net = CIFARResNextASKC(layers, cardinality, bottleneck_width, classes, use_se=False)
        model_name = 'resnext47_32x4d_aff_mixup'
    elif opt.model == 'se_resnext38_32x4d':
        num_layers = 38
        layer = (num_layers - 2) // 9
        layers = [layer] * 3
        cardinality = 32
        bottleneck_width = 4
        net = CIFARResNextASKC(layers, cardinality, bottleneck_width, classes, use_se=True)
        model_name = 'se_resnext38_32x4d_askc_mixup'

    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context)
    optimizer = 'nag'

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
        logging_handlers.append(logging.FileHandler('%s/train_%s_%s.log' %
                                                    (logging_dir, opt.dataset, model_name)))

    logging.basicConfig(level=logging.INFO, handlers = logging_handlers)
    logging.info(opt)

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    def label_transform(label, classes):
        ind = label.astype('int')
        res = nd.zeros((ind.shape[0], classes), ctx=label.context)
        res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1
        return res

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
            # CIFAR10
            train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
                batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
            val_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif opt.dataset == 'cifar100':
            # CIFAR100
            train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR100(train=True).transform_first(transform_train),
                batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
            val_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR100(train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            raise ValueError('Unknown Dataset')

        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.RMSE()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
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
            alpha = 1

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                lam = np.random.beta(alpha, alpha)
                if epoch >= epochs - 20:
                    lam = 1

                data_1 = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label_1 = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                data = [lam*X + (1-lam)*X[::-1] for X in data_1]
                label = []
                for Y in label_1:
                    y1 = label_transform(Y, classes)
                    y2 = label_transform(Y[::-1], classes)
                    label.append(lam*y1 + (1-lam)*y2)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                output_softmax = [nd.SoftmaxActivation(out) for out in output]
                train_metric.update(label, output_softmax)
                name, acc = train_metric.get()
                iteration += 1

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(ctx, val_data)
            train_history.update([acc, 1-val_acc])
            train_history.plot(save_path='%s/%s_history.png'%(plot_name, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-%s-%s-best.params' %
                                    (save_dir, best_val_score, opt.dataset, model_name))

            name, val_acc = test(ctx, val_data)
            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                (epoch, acc, val_acc, train_loss, time.time()-tic))

        host_name = socket.gethostname()
        with open(opt.dataset + '_' + host_name + '_GPU_' + opt.gpus + '_best_Acc.log', 'a') as f:
            f.write('best Acc: {:.4f}\n'.format(best_val_score))

    if not opt.summary:
        if opt.mode == 'hybrid':
            net.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
