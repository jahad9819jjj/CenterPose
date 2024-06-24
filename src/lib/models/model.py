from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.pose_dla_dcn import get_dla_dcn_convGRU
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net

_model_factory = {
    'res': get_pose_net,  # default Resnet with deconv
    'dlav0': get_dlav0,  # default DLAup
    'dla': get_dla_dcn,
    'dlav1': get_dla_dcn_convGRU,
    'resdcn': get_pose_net_dcn, # Not tested yet
    'hourglass': get_large_hourglass_net, # Not tested yet
}


def create_model(arch, heads, head_conv, opt=None):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, opt=opt)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    '''
    resume:学習再開ステップ
    lr_step:学習率の変更ステップ
    '''

    # 開始エポックを0に設定し指定パスからチェックポイントをロード
    # map_locationはデバイス不変のモデルとし、現在のデバイスにロードする(モデルがGPUで保存されたがCPUしか利用できない場合でもエラーなくロードできるようにする)
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    # WHY??
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            # "module."の文字列削除のため
            # nn.DataParallelを使用してモデルを複数のGPUで学習するとモデルの各層の前に"module."が自動的に追加されるため
            # 上の要因でもし複数GPUで学習したモデルを異なる環境にロードした場合に問題が発生する可能性がある
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    # get current model state dict
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    # サイズの不一致や欠落しているパラメタを確認するため
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    try:
        # For PyTorch version higher than 1.6
        torch.save(data, path, _use_new_zipfile_serialization=False)
    except:
        # For old version
        torch.save(data, path)
