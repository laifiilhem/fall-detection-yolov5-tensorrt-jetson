import sys
import argparse
import os
import struct
import torch
from pathlib import Path
from models.yolo import Model
from utils.general import check_yaml
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert state_dict .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    parser.add_argument('-c', '--cfg', required=True, help='Model config .yaml (e.g. yolov5n.yaml)')
    parser.add_argument('-n', '--nc', type=int, required=True, help='Number of classes (e.g. 6 for PCB)')
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input weights file path')
    if not os.path.isfile(args.cfg):
        raise SystemExit('Invalid YAML model config file path')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(args.output, os.path.splitext(os.path.basename(args.weights))[0] + '.wts')

    return args.weights, args.output, args.cfg, args.nc


pt_file, wts_file, cfg_file, nc = parse_args()
print(f'Generating .wts from state_dict: {pt_file}')

# Build model from config
device = select_device('cpu')
cfg = check_yaml(cfg_file)
model = Model(cfg, ch=3, nc=nc).to(device)  # ch=3 for RGB images

# Load state_dict
print(f'Loading state_dict from {pt_file}')
state_dict = torch.load(pt_file, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Export weights to .wts
print(f'Writing weights into {wts_file}')
with open(wts_file, 'w') as f:
    f.write(f'{len(model.state_dict().keys())}\n')
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write(f'{k} {len(vr)} ')
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')

print('✅ Conversion complete!')


"""
import sys
import argparse
import os
import struct
import torch
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-o', '--output', help='Output (.wts) file path (optional)')
    parser.add_argument(
        '-t', '--type', type=str, default='detect', choices=['detect', 'cls', 'seg'],
        help='determines the model is detection/classification')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args.weights, args.output, args.type


pt_file, wts_file, m_type = parse_args()
print(f'Generating .wts for {m_type} model')

# Load model
print(f'Loading {pt_file}')
device = select_device('cpu')
model = torch.load(pt_file, map_location=device)  # Load FP32 weights
model = model['ema' if model.get('ema') else 'model'].float()

if m_type in ['detect', 'seg']:
    # update anchor_grid info
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
    # model.model[-1].anchor_grid = anchor_grid
    delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
    # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
    model.model[-1].register_buffer("anchor_grid", anchor_grid)
    model.model[-1].register_buffer("strides", model.model[-1].stride)

model.to(device).eval()

print(f'Writing into {wts_file}')
with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
"""
