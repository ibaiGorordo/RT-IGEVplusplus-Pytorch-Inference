import argparse
import time

import cv2
import numpy as np
import torch

from rt_igev import IGEVStereo, InputPadder
device = 'cuda'

def load_image(imfile):
    img = cv2.imread(imfile, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./models/sceneflow.pth')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = IGEVStereo(args)
    checkpoint = torch.load(args.restore_ckpt, weights_only=True)

    # Remove module. from name since we are not using DataParallel
    for key in list(checkpoint.keys()):
        if 'module' in key:
            checkpoint[key.replace('module.', '')] = checkpoint.pop(key)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    image1 = load_image("Campus_CCW_Clear_Day_002/left.png")
    image2 = load_image("Campus_CCW_Clear_Day_002/right.png")
    padder = InputPadder(image1.shape, divis_by=32)

    baseline = 0.120  # meters
    focal_length_pixels = 640

    max_depth = 20  # meters
    with torch.no_grad():
        image1, image2 = padder.pad(image1, image2)
        start = time.perf_counter()
        for i in range(100):
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
        end = time.perf_counter()
        print(f"Time taken: {(end - start) / 100:.2f} seconds")
        disp = padder.unpad(disp)
        disp = disp.cpu().numpy().squeeze()

        # torch.onnx.export(model,
        #                   (image1, image2),
        #                   "model.onnx")

        # Convert disparity to depth
        depth = (baseline * focal_length_pixels) / disp
        depth[depth > max_depth] = max_depth

        depth = depth / max_depth
        depth = 255 - np.round(depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
        cv2.imshow("Disparity", depth_color)
        cv2.waitKey(0)
