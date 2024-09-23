import cv2
import torch
from imread_from_url import imread_from_url

from rt_igev import IGEVStereo
from rt_igev.utils import process_image, depth_from_disp, draw_depth
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    model_path = "models/sceneflow.pth"
    model = IGEVStereo(model_path)
    model.to(device)
    model.eval()

    left_img = imread_from_url("https://raw.githubusercontent.com/ibaiGorordo/SIDMini-Stereo-Dataset-Autonomous-Driving/refs/heads/main/Campus_CCW_Clear_Day_002/left.png")
    right_img = imread_from_url("https://raw.githubusercontent.com/ibaiGorordo/SIDMini-Stereo-Dataset-Autonomous-Driving/refs/heads/main/Campus_CCW_Clear_Day_002/right.png")

    left_input, _ = process_image(left_img, device)
    right_input, padder = process_image(right_img, device)

    with torch.no_grad():
        disp = model(left_input, right_input)
        disp = padder.unpad(disp)
        disp = disp.cpu().numpy().squeeze()

        # Convert disparity to depth
        depth = depth_from_disp(disp, baseline=0.12, focal_length=left_img.shape[1]/2)

        # Draw depth map
        depth_color = draw_depth(depth, max_distance=20) # 20 meters

        cv2.imshow("Disparity", depth_color)
        cv2.waitKey(0)
