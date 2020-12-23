# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        
        
class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        
        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def pull_loss(predicted_embeddings, tcls, tasc, device):
    """Returns pull loss by finding mean squared error between pairs of faces and bodies of the same person

    Args:
        predicted_embeddings (contains a tensor of size total targets generated for this image): [description]
        tcls ([type]): [description]
        tasc ([type]): [description]
    """

    predicted_embeddings_ = torch.cat(predicted_embeddings)
    tcls_ = torch.cat(tcls)
    tasc_ = torch.cat(tasc)

    person_ids = tasc_.unique()

    all_loss = torch.zeros(1, device=device)
    for id in person_ids:
        indices = torch.where(tasc_ == id)
        person_cls = tcls_[indices] # Classes from target for this person
        person_embeddings = predicted_embeddings_[indices] # Predicted embeddings for this person

        # import pdb; pdb.set_trace()

        faces_indices = torch.where(person_cls == 1)[0]
        body_indices = torch.where(person_cls == 0)[0]

        # for each body traverse all faces and get embeddings
        person_loss = torch.zeros(1, device=device)
        pair_count = 0
        for bi in body_indices:
            e_body = person_embeddings[bi]
            for fi in faces_indices:
                e_face = person_embeddings[fi]

                ek = (e_body + e_face) / 2 # ek is the average of the two embeddings
                person_loss += (e_face - ek)**2 + (e_body - ek)**2
                pair_count += 1
        
        if pair_count != 0:
            person_loss /= pair_count

        all_loss += person_loss
    
    all_loss /= person_ids.shape[0]
    
    return all_loss

def push_loss(predicted_embeddings, tcls, tasc, device, delta=1):
    # Seperate all persons
    # For each person who is not the current person
    # Traverse all pairs of body and face
    # Calculate push loss

    # import pdb; pdb.set_trace()

    predicted_embeddings_ = torch.cat(predicted_embeddings)
    tcls_ = torch.cat(tcls)
    tasc_ = torch.cat(tasc)

    person_ids = tasc_.unique()


    all_loss = torch.zeros(1, device=device)
    pair_count = 0
    for ci in person_ids:
        # Current person

        c_indices = torch.where(tasc_ == ci)
        if 0 in c_indices[0].shape:
            continue


        c_person_cls = tcls_[c_indices] # Classes from target for this person
        c_person_embeddings = predicted_embeddings_[c_indices] # Predicted embeddings for this person

        for oi in person_ids:
            # Other person
            if ci == oi:
                continue

            o_indices = torch.where(tasc_ == oi)
            if 0 in o_indices[0].shape:
                continue

            o_person_cls = tcls_[o_indices] # Classes from target for this person
            o_person_embeddings = predicted_embeddings_[o_indices] # Predicted embeddings for this person

            for c_idx, c_part in enumerate(c_person_cls):
                c_e = c_person_embeddings[c_idx]
                for o_idx, o_part in enumerate(o_person_cls):
                    if c_part == o_part:
                        continue # Skip this part because they are the same

                    pair_count += 1
                    o_e = o_person_embeddings[o_idx]

                    all_loss += max(0, delta-abs(c_e - o_e))

    d = (pair_count * (pair_count - 1))
    if d != 0:
        all_loss /= d

    return all_loss


def compute_loss(p, targets, model, assocs):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    lpull, lpush = torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, tasc = build_targets(p, targets, model, assocs)  # targets

    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Add another association loss here

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    predicted_embeddings = []

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Pull loss
            # ps contains the predictions which correspond to the generated targets
            # target classes are available in tcls[i]
            # associations are available in 
            # we need to concatenate all ps's into a long n,8 vector which will then be used to create push pull loss for associations
            # traverse all pairs of the same person over all scales
            # get ps[7] which is the embedding
            # for all pairs calculate the following (((ef - ek)**2 + (ep - ek)**2) * 1/npairs) * 1/n_persons

            # Push loss
            predicted_embeddings.append(ps[:, 7])

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    if targets.shape[0]:
        lpull += pull_loss(predicted_embeddings, tcls, tasc, device)
        lpush += push_loss(predicted_embeddings, tcls, tasc, device)

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls + lpull + lpush
    return loss * bs, torch.cat((lbox, lobj, lcls, loss, lpull, lpush)).detach()


def build_targets(p, targets, model, assocs):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets

    # tcls contains the classes per anchor per  
    tcls, tbox, indices, anch, tasc = [], [], [], [], []

    temb = []
    gain = torch.ones(7 + 1, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

    # Appending associations directly to the targets here        
    targets = torch.cat((targets, assocs.view(-1,1)), 1)

    # Repeat the targets 3 times for each indices and append the anchor pair indices to it for reference
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]

        # Gain is used to denormalise the targets to the shape of the detect scale at which we are comparing to anchors
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        # Targets are denormalised here
        t = targets * gain

        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # The above does this in short
                # width height ratio is calculated between all anchors at the given scale.
                # If the widht to height of vice versa ratio is higher than the anchor threshold j for that index contains a True
                # t at the end contains only targets which fit the given anchors at that scale.
                # For example if you have 6 ground truths (6 x 6 matrix)
                    # t would at this point be 3,6,6+1 where the gt is repeated and anchor index is appended to each ground truth
                    # Once filtered t could look like the following  18,6 if all gt boxes were less than anchor_t ratio between the width and height 

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            # Append associations to T
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        asc = t[:, 6].long() # association numbers
        a = t[:, 7].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        tasc.append(asc)


    return tcls, tbox, indices, anch, tasc
