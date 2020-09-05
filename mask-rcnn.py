import os
import numpy as np
import torch
from PIL import Image
import glob
import utils
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import evaluate
from engine import train_one_epoch, evaluate
from albumentations import (HorizontalFlip, ShiftScaleRotate, VerticalFlip, Normalize,Flip,Compose, GaussNoise)
from albumentations.pytorch import ToTensor

def get_transforms(phase):
  list_transforms = []
  if phase == 'train':
    list_transforms.extend([Flip(p=0.5)])
  list_transforms.extend([ToTensor(),])
  list_trfms = Compose(list_transforms,bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
  return list_trfms

class LOAD_DATASET(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.transforms = get_transforms(phase)
        self.phase = phase
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "IMAGE"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "MASK"))))
        self.classes = {
                            "TRAIN CALC BENIGN":    [1,(255,0,0)],
                            "TRAIN CALC MALIGNANT": [2,(0,0,255)],
                            "TRAIN MASS BENIGN":    [3,(0,255,0)],
                            "TRAIN MASS MALIGNANT": [4,(255,0,100)],

                            "TEST CALC BENIGN":     [1,(255,0,0)],
                            "TEST CALC MALIGNANT":  [2,(0,0,255)],
                            "TEST MASS BENIGN":     [3,(0,255,0)],
                            "TEST MASS MALIGNANT":  [4,(255,0,100)],
                        }
                        

    # def get_transform(self, image, mask):
    #     # Resize
    #     resize = T.Resize(size=(520, 520))
    #     image = resize(image)
    #     mask = resize(mask)

    #     # Random crop
    #     i, j, h, w = T.RandomCrop.get_params(
    #         image, output_size=(512, 512))
    #     image = TF.crop(image, i, j, h, w)
    #     mask = TF.crop(mask, i, j, h, w)

    #     # Random horizontal flipping
    #     if random.random() > 0.5:
    #         image = TF.hflip(image)
    #         mask = TF.hflip(mask)
    #     # Random vertical flipping
    #     if random.random() > 0.5:
    #         image = TF.vflip(image)
    #         mask = TF.vflip(mask)

    #     # Transform to tensor
    #     image = TF.to_tensor(image)
    #     mask = TF.to_tensor(mask)
        
    #     return image, mask

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "IMAGE", self.imgs[idx])
        mask_path = os.path.join(self.root, "MASK", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        
        trans = T.Compose([
            T.ToTensor()
        ])
    
        mask = Image.open(mask_path)
        mask_origin = mask.copy()
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # if xmin<0 or ymin<0 or xmax<0 or ymax<0:
        #     print(imgs[idx])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.tensor(self.classes[(self.imgs[idx].split("_")[0])][0],dtype=torch.int64)
        
        masks = torch.as_tensor(masks, dtype=torch.int)

        

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor(idx)
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        # if self.transforms:
        sample = {
            'image': img,
            'bboxes': target['boxes'],
            'labels': target['labels']
        }
        sample = self.transforms(**sample)
        image = sample['image']
            
        target['boxes'] = torch.stack(tuple(map(torch.tensor, 
                                                zip(*sample['bboxes'])))).permute(1, 0)
        
        return image, target


    def __len__(self):
        return len(self.imgs)

dataset = LOAD_DATASET('../DATA TRAIN/TRAIN/', "train")
dataset_test = LOAD_DATASET('../DATA TRAIN/TRAIN/', "test")


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=8,
    collate_fn=utils.collate_fn)

"""
    load model
"""


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

"""
    define model
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# dataset has two classes only - background and person
num_classes = 5

model = get_instance_segmentation_model(num_classes).to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=0.00001)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=25,
                                               gamma=0.1)


num_epochs = 100

for epoch in range(num_epochs):
    
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    
    # update the learning rate
    lr_scheduler.step()
    
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
