import argparse
import time

import cv2
import numpy as np
import torch

from rt_igev import IGEVStereo, InputPadder
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    img = cv2.imread(imfile, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

if __name__ == '__main__':
    model_path = "models/sceneflow.pth"
    model = IGEVStereo(model_path)
    model.to(device)
    model.eval()

    image1 = load_image("Campus_CCW_Clear_Day_002/left.png")
    image2 = load_image("Campus_CCW_Clear_Day_002/right.png")
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)
    disp = model(image1, image2) # Warmup

    baseline = 0.120  # meters
    focal_length_pixels = 640

    max_depth = 20  # meters
    n_tests = 10
    with torch.no_grad():
        start = time.perf_counter()
        for i in range(n_tests):
            disp = model(image1, image2)
        end = time.perf_counter()
        print(f"Time taken: {(end - start)/n_tests:.2f} seconds")
        disp = padder.unpad(disp)
        disp = disp.cpu().numpy().squeeze()

        # torch.onnx.export(model,
        #                   (image1, image2),
        #                   "model.onnx",
        #                   opset_version=16,
        #                   )

        # Convert disparity to depth
        depth = (baseline * focal_length_pixels) / disp
        depth[depth > max_depth] = max_depth

        depth = depth / max_depth
        depth = 255 - np.round(depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
        cv2.imshow("Disparity", depth_color)
        cv2.waitKey(0)
