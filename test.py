import argparse
import glob
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, find_target, \
    non_max_suppression, scale_coords, score_pairs, xyxy2xywh, xywh2xyxy, set_logging, increment_path, score_heuristically, score_pairs, find_target
from utils.loss import compute_loss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0):  # number of logged images

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(5, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    assoc_stats = {
        "tp":[0]*iouv.shape[0],
        "tn":[0]*iouv.shape[0],
        "fp":[0]*iouv.shape[0],
        "fn":[0]*iouv.shape[0],
        "total_associations":[0]*iouv.shape[0],
        "total":[0]*iouv.shape[0]
    }
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

        # if batch_i > 200:
        #     break

        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:
                int_loss = compute_loss([x.float() for x in train_out], targets, model, compute_embedding_loss=True)[1]  # box, obj, cls
                int_loss = torch.cat((int_loss[:3], int_loss[-2:]))
                loss += int_loss

            # Run NMS
            targets[:, 2:6] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_txt else []  # for autolabelling
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))


                # Per target class
                # targets_for_predictions = torch.zeros(pred.shape[0], targets.shape[0], dtype=torch.bool, device=device) # for a target traversed py prediction set to true
                targets_for_predictions = torch.zeros((pred.shape[0], targets.shape[0], iouv.shape[0]), dtype=torch.bool, device=device) # for a target traversed py prediction set to true

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1) # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # Finding the ious of all target boxes against all predictions and finding max in 1st axis. 
                        # And the maximum IOU is filtered for each prediciton box.
                        # i contains the target box that most overlaps with the prediction box at that index
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices.

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False): # traversing predictions which have high overlap
                            # Targets which have above a threshold overlap are traveresed here
                            d = ti[i[j]]  # detected target. i is the list of targets with max overlap for the prediction at each index
                            
                            # J contains predictions and I contains targets. 
                            # i[j] is the target regressed by the prediction j. For prediction j we can have a list of targets traversed
                            targets_for_predictions[j, i[j], :] = ious[j] > iouv # For each prediction, target pair we have a set of iou pass

                            # if a target has not yet been detected only then do we traverse itself
                            # Otherwise we skip this current index of prediction which 
                            if d.item() not in detected_set:
                                detected_set.add(d.item()) # The target is marked as already assigned
                                
                                # Which prediction regresses this target though
                                # Lets store this info as well in an array

                                detected.append(d) 
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn. The correct array marks the predictions if the preds regress the target at progressively increasing thresholds 
                                if len(detected) == nl:  # all targets already located in image
                                    # Break if all targets are found by a prediction box
                                    break

                # Note that predn[:, -1] contains the embedding vectors.
                # Only for the predictions in the detected set we score the embedding vectors and 
                # produce pairs. Then we see if the target bodies have the same faces as the one predicted.
                # And a simple correct count/pair count should produce accuracy
                
                # Score both methods though. The heuristical one and the other one as well. 
                # For that we need to run both algos here as well
                

                # We need to traverse the iouv first
                # For all iouv
                    # compute output (from both methods) for correct pairs at this iouv level
                    # Now we know which face has which body for the predictions
                    # And we know which prediction associated with which target. 
                    # from output traverse all pairs
                    # for the prediction pair find targets from the targets_for_prediction array
                body_label = 0.
                face_label = 1.
                for idx, iou in enumerate(iouv):
                    correct_preds = [predn[correct[:, idx]]]
                    output_embedding = score_pairs(correct_preds)[0]
                    # output_heuristic = score_heuristically(correct_preds)[0]

                    
                    # ==================== OLD ACCURACIES
                    # all_faces = output_embedding[output_embedding[:, -2] == face_label]
                    # nassoc = (all_faces[:, -1] != -1).nonzero(as_tuple=False).flatten()
                    # assoc_stats["total"][idx] += len(nassoc)

                    # # First our method
                    # for jidx in (output_embedding[:, -2] == face_label).nonzero(as_tuple=False):
                    #     # Find associated body
                    #     pred_face = output_embedding[jidx]
                    #     pred_body_ix = int(output_embedding[jidx, -1].item())
                    #     pred_body = None

                    #     # Checking if Body is associated with the predicted face
                    #     if pred_body_ix != -1:
                    #         pred_body = output_embedding[pred_body_ix]

                    #         # ======= Finding best target body.
                    #         ti_body = (tcls_tensor == torch.tensor(body_label)).nonzero(as_tuple=False).view(-1) # target indices
                    #         ious, i = box_iou(pred_body[:4].unsqueeze(0), tbox[ti_body]).max(1)  # best ious, indices.

                    #         best_body_target = None
                    #         if (ious > iou).item():
                    #             best_body_target = targets[ti_body[i]]

                    #             assoc_face_ix = ((targets[:, 1] == face_label) & (targets[:, -1] == best_body_target[:, -1])).nonzero(as_tuple=False).flatten()
                    #             assoc_face = None
                    #             if 0 in assoc_face_ix.shape:
                    #                 # No associated face for the target
                    #                 pass
                    #             else:
                    #                 assoc_face = targets[assoc_face_ix]

                    #     # ======= Find associated face
                    #     ti_face = (tcls_tensor == torch.tensor(face_label)).nonzero(as_tuple=False).view(-1) # target indices
                    #     ious, i = box_iou(pred_face[:, :4], tbox[ti_face]).max(1)  # best ious, indices.

                    #     best_face_target_ix = ti_face[i]

                    #     # For each iouv we need to store 
                    #     # Correct associations, True positive
                    #     # True negative, body doesnt have face and was not associated with face
                    #     # False positive, body had face but no association was made
                    #     # False negative, body doesnt have face but was associated with face
                    #     # Top-3 association -- optional
                    #     if type(assoc_face) != type(None) and assoc_face_ix == best_face_target_ix:
                    #         # This was a correct association from the embedding technique
                    #         assoc_stats["tp"][idx] += 1
                    # ====================

                    all_bodies = output_embedding[output_embedding[:, -2] == body_label]
                    nassoc = (all_bodies[:, -1] != -1).nonzero(as_tuple=False).flatten()
                    assoc_stats["total"][idx] += all_bodies.shape[0] # The number of associations include the associations that were not made because they also amount to the accuracy
                    assoc_stats["total_associations"][idx] += len(nassoc) # The number of associations include the associations that were not made because they also amount to the accuracy

                    for jidx in (output_embedding[:, -2] == body_label).nonzero(as_tuple=False):

                        # Predicted body
                        pred_body = output_embedding[jidx].flatten()
                        # print("This is the predicted body: ", pred_body)

                        # Find target body
                        target_body_bb, target_body_ix, target_body = find_target(tcls_tensor, tbox, pred_body, body_label, iou, targets)
                        # print("This is the target body: ", target_body)

                        # Predicted Associated Face
                        pred_face_ix = int(output_embedding[jidx, -1].item())
                        pred_face = None
                        if pred_face_ix != -1:
                            pred_face = output_embedding[pred_face_ix].flatten()
                            # print("This is the (associated )predicted face: ", pred_face)
                        else: 
                            # There is no associated predicted face
                            pass

                        # Target Associated Face. Face associated with the target body
                        target_face_ix = -1
                        target_face = None
                        if type(target_body) != type(None):
                            _ix = ((targets[:, 1] == face_label) & (targets[:, -1] == target_body[:, -1])).nonzero(as_tuple=False).flatten()
                            if len(_ix) > 0:
                                # If there was a target body and it had an association we get it here
                                target_face_ix = _ix
                                target_face = targets[target_face_ix] # Found target face
                        else:
                            # No target body found
                            # And this has no effect on the association accuracy
                            # So we can just break. Because no target was found
                            break
                        
                        # Correct associations, True positive
                        # True negative, body doesnt have face and was not associated with face
                        # False positive, body had face but no association was made 
                        # False negative, body doesnt have face but was associated with face

                        # Firstly if we have target face and predicted face check if they are the same 
                        if type(pred_face) != type(None) and type(target_face) != type(None):
                            # We have both faces
                            # Finding the target face
                            target_face_alt_bb, target_face_alt_ix, target_face_alt = find_target(tcls_tensor, tbox, pred_face, face_label, iou, targets)

                            # print(target_face_ix)
                            # print(target_face_alt_ix)

                            if target_face_alt_ix == target_face_ix:
                                # This means both the target box regressed by the prediction and the target face are the same
                                assoc_stats["tp"][idx] += 1
                            else:
                                # The faces are not the same. Which means the association is wrong
                                # This makes the association a false positive
                                assoc_stats["fp"][idx] += 1
                        
                        if type(pred_face) == type(None) and type(target_face) == type(None):
                            # No predicted face from the body and no target face from the target either
                            assoc_stats["tp"][idx] += 1
                        
                        if type(target_face) != type(None) and type(pred_face) == type(None):
                            # body had face but no association was made 
                            assoc_stats["fp"][idx] += 1

                        if type(target_face) == type(None) and type(pred_face) != type(None):
                            # if we have a target face from the target body but no prediction was made.
                            assoc_stats["fn"][idx] += 1

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    print(assoc_stats)

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = glob.glob('../coco/annotations/instances_val*.json')[0]  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(f, x)  # plot
