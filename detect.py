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
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt,plot_one_box


device = '0'
poseweights = 'yolov7-w6-pose.pt'
source = 'videos/test.mp4'
hide_labels = False
hide_conf = False
line_thickness = 4
view_img = True
save = False

steps = 3
radius = 5
counter = 20
degriment = False
x_9, y_9, x_10, y_10 = 0, 0, 0, 0

def plot_skeleton_kpts(im, kpts, steps=3, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    print(len(palette))
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


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
            # frame_roi = frame[650:900, 30:350]
            x_9, y_9, x_10, y_10 = 0, 0, 0, 0
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
                    # for c in pose[:, 5].unique(): # Print results
                    #     n = (pose[:, 5] == c).sum()  # detections per class
                    #     print("No of Objects in Current Frame : {}".format(n))
                    
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                        # c = int(cls)  # integer class
                        kpts = pose[det_index, 6:]
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        # Object Bounding Box
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(im0, c1, c2, (0, 255, 0), 2)
                        # Pose Landmarks
                        # plot_skeleton_kpts(im0, kpts, 3)
                        num_kpts = len(kpts) // steps
                        for kid in range(num_kpts):
                            r, g, b = [0, 255, 255]
                            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
                            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                                if steps == 3:
                                    conf = kpts[steps * kid + 2]
                                    if conf < 0.5:
                                        continue
                                cv2.circle(im0, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                                cv2.putText(
                                    im0, f'{kid}', (int(x_coord), int(y_coord)), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (0, 0, 255), 3
                                )
                                if kid == 9:
                                    x_9, y_9 = int(x_coord), int(y_coord)
                                if kid == 10:
                                    x_10, y_10 = int(x_coord), int(y_coord)
                                if x_9 > 0 and x_10 > 0:
                                    if (30<x_9<350 and 650<y_9<900) or (30<x_10<350 and 650<y_10<900):
                                        # degriment = True
                                        counter -= 1

            
            # Degrement
            # if degriment:
            #     counter -= 1
            #     degriment = False

            if counter > 8:
                plot_one_box([30, 650, 350, 900], im0, (0,250,0), f'Count: {counter}', 3)
            else:
                plot_one_box([30, 650, 350, 900], im0, (0,0,250), f'Count: {counter}', 3)

            # cv2.rectangle(
            #     im0,  (30, 650), (350, 900), (255, 255, 0), 3
            # )

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
