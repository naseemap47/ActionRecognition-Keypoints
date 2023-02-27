import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt


device = '0'
poseweights = 'yolov7-w6-pose.pt'
source = 'videos/video1.mp4'
hide_labels = False
hide_conf = False
line_thickness = 4
view_img = True
save = False

steps = 2
radius = 5
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

device = select_device(device)
half = device.type != 'cpu'

model = attempt_load(poseweights, map_location=device)
_ = model.eval()
names = model.module.names if hasattr(model, 'module') else model.names

if source.isnumeric() :    
    cap = cv2.VideoCapture(int(source))
else:
    cap = cv2.VideoCapture(source)

if (cap.isOpened() == False):   #check if videocapture not opened
    print('Error while trying to read video. Please check path again')
    raise SystemExit()

else:

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))        
    if save:
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

    while(cap.isOpened):

        ret, frame = cap.read()
        
        if ret:
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
        
            image = image.to(device)
            image = image.float()
        
            with torch.no_grad():
                output_data, _ = model(image)

            output_data = non_max_suppression_kpt(output_data,
                                        0.25,   # Conf. Threshold.
                                        0.65, # IoU Threshold.
                                        nc=model.yaml['nc'], # Number of classes.
                                        nkpt=model.yaml['nkpt'], # Number of keypoints.
                                        kpt_label=True)
        
            output = output_to_keypoint(output_data)

            im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
            im0 = im0.cpu().numpy().astype(np.uint8)
            
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for i, pose in enumerate(output_data):  # detections per image
            
                if len(output_data):  #check if no pose
                    for c in pose[:, 5].unique(): # Print results
                        n = (pose[:, 5] == c).sum()  # detections per class
                        print("No of Objects in Current Frame : {}".format(n))
                    
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                        c = int(cls)  # integer class
                        kpts = pose[det_index, 6:]
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                        #             line_thickness=line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                        #             orig_shape=im0.shape[:2])
                        # print(kpts)

                        # Pose Landmarks
                        
                        num_kpts = len(kpts) // steps
                        for kid in range(num_kpts):
                            # r, g, b = pose_kpt_color[kid]
                            r, g, b = [0, 255, 255]
                            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
                            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                                if steps == 3:
                                    conf = kpts[steps * kid + 2]
                                    if conf < 0.5:
                                        continue
                                cv2.circle(im0, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                                cv2.putText(im0, f'{kid}', (int(x_coord), int(y_coord)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            # Stream results
            if view_img:
                cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if save:
                out.write(im0)

        else:
            break

    cap.release()
    if view_img:
        cv2.destroyAllWindows()
    if save:
        out.release()
