import numpy as np
import torch
import torch.nn.functional as F
import cv2

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    # print("$$$55555", img.shape, coords.shape)
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1

    # print("######88888", xgrid)
    # assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # print("###37777", grid.shape)
    img = F.grid_sample(img, grid, align_corners=True)

    return img

def process_image(img, device, divis_by=32):
    input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input = torch.from_numpy(input).permute(2, 0, 1).float()
    padder = InputPadder(input.shape, divis_by=divis_by)
    input = padder.pad(input[None])[0].to(device)
    return input, padder

def depth_from_disp(disp, baseline, focal_length):
    return baseline * focal_length / (disp + 1e-9)

def draw_depth(depth_map, max_distance):
    depth = np.clip(depth_map, 0, max_distance)
    depth = depth / max_distance
    depth = 255 - np.round(depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)