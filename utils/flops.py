# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com
from collections import OrderedDict
from mxnet import ndarray
from mxnet.gluon.nn import HybridBlock


def summary(block, *inputs):
    """Print the summary of the model's output and parameters.
    The network must have been initialized, and must not have been hybridized.
    Parameters
    ----------
    inputs : object
        Any input that the model supports. For any tensor in the input, only
        :class:`mxnet.ndarray.NDArray` is supported.
    """
    summary = OrderedDict()
    seen = set()
    hooks = []

    def _get_shape_str(args):
        def flatten(args):
            if not isinstance(args, (list, tuple)):
                return [args], int(0)
            flat = []
            fmts = []
            for i in args:
                arg, fmt = flatten(i)
                flat.extend(arg)
                fmts.append(fmt)
            return flat, fmts

        def regroup(args, fmt):
            if isinstance(fmt, int):
                if fmt == 0:
                    return args[0], args[1:]
                return args[:fmt], args[fmt:]
            ret = []
            for i in fmt:
                res, args = regroup(args, i)
                ret.append(res)
            return ret, args

        flat_args, fmts = flatten(args)
        flat_arg_shapes = [x.shape if isinstance(x, ndarray.NDArray) else x
                           for x in flat_args]

        shapes = regroup(flat_arg_shapes, fmts)[0]

        if isinstance(shapes, list):
            shape_str = str(shapes)[1:-1]
        else:
            shape_str = str(shapes)
        return shape_str.replace('L', '')

    def _flops_str(flops):
        preset = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]

        for p in preset:
            if flops // p[0] > 0:
                N = flops / p[0]
                ret = "%.1f%s" % (N, p[1])
                return ret
        ret = "%.1f" % flops
        return ret

    def _calculate_conv2d_flops(block, output):
        flops = 0
        o_w = output[2]
        o_h = output[3]
        for i, p in enumerate(block.params.values()):
            # weight
            if i == 0:
                weisht_shape = p.data().shape
                o_c = weisht_shape[0]
                i_c = weisht_shape[1]
                ker_w = weisht_shape[2]
                ker_h = weisht_shape[3]
                groups = block._kwargs['num_group']
                flops += i_c * ker_h * ker_w * o_c * o_w * o_h / groups
            # bias
            elif i == 1:
                bias_shape = p.data().shape[0]
                flops += bias_shape * o_h * o_w
            else:
                raise NotImplementedError
        return flops

    def _calculate_dense_flops(block):
        # print(block.params.values())
        flops = 0
        for i, p in enumerate(block.params.values()):
            # weight
            if i == 0:
                weisht_shape = p.data().shape
                flops += 2 * weisht_shape[0] * weisht_shape[1] - weisht_shape[1]
            # bias
            elif i == 1:
                flops += p.data().shape[0]
            else:
                raise NotImplementedError
        return flops

    def _register_summary_hook(block):
        assert not isinstance(block, HybridBlock) or not block._active, \
            '"{}" must not be hybridized to print summary.'.format(block.name)

        def _summary_hook(block, inputs, outputs):
            class_name = block.__class__.__name__
            block_idx = len(summary) - 1

            m_key = '%s-%i' % (class_name, block_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['output_shape'] = _get_shape_str(outputs)

            params = 0
            summary[m_key]['trainable'] = 0
            summary[m_key]['shared'] = 0
            for p in block.params.values():
                params += p.data().size
                summary[m_key]['trainable'] += 0 if p.grad_req == 'null' else p.data().size
                if p in seen:
                    summary[m_key]['shared'] += p.data().size
                else:
                    seen.add(p)
            summary[m_key]['n_params'] = params

            flops = 0
            if class_name == 'Conv2D':
                flops += _calculate_conv2d_flops(block, outputs.shape)
            elif class_name == 'Dense':
                flops += _calculate_dense_flops(block)
            else:
                pass
            summary[m_key]['n_flops'] = int(flops)

        from mxnet.gluon.nn.basic_layers import Sequential, HybridSequential
        if not isinstance(block, (Sequential, HybridSequential)):
            hooks.append(block.register_forward_hook(_summary_hook))

    summary['Input'] = OrderedDict()
    summary['Input']['output_shape'] = _get_shape_str(inputs)
    summary['Input']['n_flops'] = 0
    summary['Input']['n_params'] = 0
    summary['Input']['trainable'] = 0
    summary['Input']['shared'] = 0

    try:
        block.apply(_register_summary_hook)
        block(*inputs)

        line_format = '{:>20}  {:>42} {:>15} {:>15}'
        print('-' * 96)
        print(line_format.format('Layer (type)', 'Output Shape', 'FLOPs', 'Param #'))
        print('=' * 96)
        total_flops = 0
        total_params = 0
        trainable_params = 0
        shared_params = 0
        for layer in summary:
            print(line_format.format(layer,
                                     str(summary[layer]['output_shape']),
                                     summary[layer]['n_flops'],
                                     summary[layer]['n_params']))
            total_flops += summary[layer]['n_flops']
            total_params += summary[layer]['n_params']
            trainable_params += summary[layer]['trainable']
            shared_params += summary[layer]['shared']
        print('=' * 96)
        print('Parameters in forward computation graph, duplicate included')
        print('   Total FLOPs: ' + str(total_flops) + "  " + _flops_str(total_flops))
        print('   Total params: ' + str(total_params))
        print('   Trainable params: ' + str(trainable_params))
        print('   Non-trainable params: ' + str(total_params - trainable_params))
        print('Shared params in forward computation graph: ' + str(shared_params))
        print('Unique parameters in model: ' + str(total_params - shared_params))
        print('-' * 80)
    finally:
        for h in hooks:
            h.detach()
