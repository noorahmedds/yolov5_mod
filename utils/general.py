# General utils

import glob
import logging
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import math
import numpy as np
import torch
import torchvision
import yaml

from utils.google_utils import gsutil_getsize
from utils.metrics import fitness
from utils.torch_utils import init_torch_seeds

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def check_git_status():
    # Suggest 'git pull' if repo is out of date
    if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def check_dataset(dict):
    # Download dataset if not found locally
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

from scipy.spatial import distance

def score_heuristically(prediction, body_label = 0, face_label = 1, avg_iou = 0.1273553777409504, std_iou = 0.13013710033012083, avg_vector=[0.10938075, 0.03918036]):
    output = []

    iou_thres = 0.025 # The range in which a face should normally lie for a person. This is hand featured by determining std of facial IOU with the body. 
    iou_thres_dr = avg_iou + (2*std_iou) - 0.025

    if len(prediction) != 0:
        avg_vector = torch.tensor([avg_vector], device=prediction[0].device)
    alpha = 0.6 # weight multiple for the vector dissimilarity scores
    beta = 1 - alpha # weight multiple for the iou similarity scores
    cos = torch.nn.CosineSimilarity(dim = 1)

    for pred in prediction:
        b_emb = pred[pred[:, -2] == body_label][:, -1, None]
        f_emb = pred[pred[:, -2] == face_label][:, -1, None]

        if 0 in f_emb.shape or 0 in b_emb.shape:
            break

        # Sort the predictions by cls label. face first and body next
        # pred = torch.cat((pred[pred[:, -2] == body_label], 
        #                 pred[pred[:, -2] == face_label]), 0)
        pred = torch.cat((pred[pred[:, -2] == body_label], 
                        pred[pred[:, -2] == face_label]), 0)


        # Mathcing with IOU and Relative Position ==============
        # Traverse all body indices
        # For each body find the faces with an IOU atleast greater that 0.025
        # Now sort the filtered faces by IOU again. Sort the faces by their IOU distances to (0.1273553777409504, found over dataset)
        # Also sort the faces by relative position. Sort the faces by their relative distance from [0.10938075 0.03918036]*gain
        bodies = pred[pred[:, -2] == body_label]
        faces = pred[pred[:, -2] == face_label]
        assoc_faces_set = torch.ones(faces.shape[0]).bool()

        n_body = bodies.shape[0]
        for bix, body in enumerate(bodies):
            ious = box_iou(body[None, :4], faces[:, :4])
            # filtered_indices = list(torch.where(ious > iou_thres_range[0]))
            filtered_indices = list(torch.where(abs(ious - iou_thres - (iou_thres_dr/2)) <= iou_thres_dr/2)) # Keeping between the iou thresh range

            # TODO: Filter also the faces which have already been associated
            filtered_indices[1] = filtered_indices[1][assoc_faces_set[filtered_indices[1]] == True]
            filtered_indices[0] = torch.zeros_like(filtered_indices[1])

            filtered_ious = ious[filtered_indices]

            if 0 in filtered_indices[1].shape:
                # import pdb; pdb.set_trace()
                pred[bix, -1] = -1
                continue

            if 0 in filtered_ious.shape:
                pred[bix, -1] = -1
                continue
            
            # scored_ious = torch.abs(filtered_ious - avg_iou).sort().indices # Sorted face indices by distance from avg iou

            # # Need to sort these relative vectors by dissimilarity to the avg vector
            scored_ious = torch.abs(filtered_ious - avg_iou) # Sorted face indices by distance from avg iou
            # Need to sort these relative vectors by dissimilarity to the avg vector
            relative_vectors = (faces[filtered_indices[1], :2] - body[:2]) / body[2:4] # Relative vector
            scored_cs = (1 - cos(avg_vector, relative_vectors))

            total_score = (alpha*scored_ious) + (beta*scored_cs)
            # Now we have scored the filtered faces we need to find a cummulatic score. This will be their positions in the scored tensors
            best_index = total_score.sort().indices[0]
            final_face = filtered_indices[1][best_index]
            
            pred[bix, -1] = final_face + n_body

            assoc_faces_set[final_face] = False

            # import pdb; pdb.set_trace()
            # Mathcing with IOU ==============

        output.append(pred)
    return output

def score_pairs(prediction, body_label = 0, face_label = 1):
    output = []

    for pred in prediction:
        b_emb = pred[pred[:, -2] == body_label][:, -1, None]
        f_emb = pred[pred[:, -2] == face_label][:, -1, None]

        if 0 in f_emb.shape or 0 in b_emb.shape:
            break

        # import pdb; pdb.set_trace()
        pred = torch.cat((pred[pred[:, -2] == body_label]
                        ,pred[pred[:, -2] == face_label]), 0)

        # import pdb; pdb.set_trace()
        dist_matrix = distance.cdist(b_emb.cpu(), f_emb.cpu())
        # _shape = dist_matrix.shape

        sorted_ind = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape) # Rows against columns
        face_indices = torch.where(pred[:, -2] == face_label)[0]

        # Efficient matching ==============        
        # sorted_first_indexes = np.unique(sorted_ind[1], return_index=True)[1]
        # # pair_indices = sorted_ind[0][sorted_first_indexes], sorted_ind[1][sorted_first_indexes]
        # # pair_scores = dist_matrix[pair_indices]
        # # TODO: Make sure that associated_bodies are unqiue. If they arent. We have to remove the face from the associated body
        # associated_bodies = sorted_ind[0][sorted_first_indexes]
        # pred[face_indices, -1] = torch.tensor(associated_bodies, device=pred.device).float()
        # Efficient matching ==============

        # Inefficient matching ==============
        # associated_bodies = torch.ones_like(pred[face_indices, -1]) * -1
        # seen_body = set()
        # seen_face = set()
        # for body, face in zip(*sorted_ind):
        #     if face not in seen_face:
        #         seen_face.add(face)
        #         if body not in seen_body:
        #             seen_body.add(body)
        #             associated_bodies[face] = body

        # pred[face_indices, -1] = associated_bodies
        # Inefficient matching ==============

        # Mathcing with IOU ==============
        # Traverse all face indices
        # For that face find the sorted distances
        # If with the body pair the face has an IOU choose that body as the associated body and break
        # Add the body to the seen body set
        associated_bodies = torch.ones_like(pred[face_indices, -1]) * -1
        body_count, face_count = dist_matrix.shape
        seen_body = set()
        for f in range(face_count):
            sorted_bodies = sorted_ind[0][np.where(sorted_ind[1] == f)]
            
            # Filter the bodies with an IOU with the face bounding box
            curr_face = pred[body_count + f]
            for body_idx in sorted_bodies:
                curr_body = pred[body_idx]

                if body_idx not in seen_body and  box_iou(curr_face[None, :4], curr_body[None, :4]) > 0:
                    associated_bodies[f] = body_idx
                    seen_body.add(body_idx)

        pred[face_indices, -1] = associated_bodies
        # Mathcing with IOU ==============

        output.append(pred)
    
    return output


def associate_predictions(prediction, body_label = 0, face_label = 1, delta = 10):
    """Performs associations between faces and bodies based on the embedding vector

    Returns:
        detections wih shape: nx7 (x1, y1, x2, y2, conf, cls, association_index) and also sorted by body
    """

    # Each detection will have an association index. No two bodies or faces can have the same association index
    # A face cannot be associated to two bodies
    # A body may not have a face
    output = []

    for pred in prediction:
        b_emb = pred[pred[:, -2] == body_label][:, -1, None]
        f_emb = pred[pred[:, -2] == face_label][:, -1, None]

        if 0 in f_emb.shape or 0 in b_emb.shape:
            break
        # import pdb; pdb.set_trace()
        pred = torch.cat((pred[pred[:, -2] == body_label]
                        ,pred[pred[:, -2] == face_label]), 0)

        # import pdb; pdb.set_trace()
        dist_matrix = distance.cdist(b_emb.cpu(), f_emb.cpu())

        # Because a face has to be associated with a body. We choose to minimise the face column. 
        # Now for f[0] the best body is argmin(b_emb[0])
        associated_bodies = set()

        face_idx = 0
        for i, p in enumerate(pred):
            if p[-2] == body_label:
                continue

            min_body_idx = np.argmin(dist_matrix[:, face_idx])

            if min_body_idx not in associated_bodies: 
                p[-1] = min_body_idx
                associated_bodies.add(min_body_idx)
            else:
                p[-1] = -1

            face_idx += 1

        output.append(pred)

    return output

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx7 (x1, y1, x2, y2, conf, cls, embedding)
    """

    nc = prediction[0].shape[1] - (5 + 1)  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference # bacth wise
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:-1] *= x[:, 4:5]  # conf = cls_conf * obj_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:-1] > conf_thres).nonzero(as_tuple=False).T # Non zero indexes per column
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), x[i, -1, None]), 1) # *bbox, class_score, class_label, embedding 
            # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1) # *bbox, class_score, class_label, embedding 

        else:  # best class only
            conf, j = x[:, 5:-1].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def strip_optimizer(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
