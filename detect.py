import numpy as np
import cv2


def _collect_scales(primary_size, multiscale, extra_scales):
    """Return the set of inference scales to run."""
    scales = []
    if primary_size:
        scales.append(int(primary_size))
    if multiscale and extra_scales:
        for scale in extra_scales:
            if scale is None:
                continue
            scale = int(scale)
            if scale not in scales:
                scales.append(scale)
    return scales or [960]


def _apply_nms(boxes, scores, score_thresh, iou_thresh):
    if boxes.size == 0:
        return np.empty((0, 4)), np.array([])
    xywh_boxes = []
    for x1, y1, x2, y2 in boxes:
        xywh_boxes.append(
            [
                float(x1),
                float(y1),
                max(0.0, float(x2 - x1)),
                max(0.0, float(y2 - y1)),
            ]
        )
    indices = cv2.dnn.NMSBoxes(xywh_boxes, scores.tolist(), score_thresh, iou_thresh)
    if len(indices) == 0:
        return np.empty((0, 4)), np.array([])
    keep = np.array(indices).reshape(-1)
    return boxes[keep], scores[keep]


# Task detect image
def detectImg(model, img, conf=0.5, iou_thresh=0.45, imgsz=960, multiscale=True, aux_scales=None):
    scales = _collect_scales(imgsz, multiscale, aux_scales)
    collected_boxes = []
    collected_scores = []

    for scale in scales:
        result = model(img, imgsz=scale, iou=iou_thresh)
        boxes = result[0].boxes
        conf_detect = boxes.conf.cpu().numpy()
        box_detect = boxes.xyxy.cpu().numpy()
        idx = np.where(conf_detect >= conf)[0]
        if idx.size == 0:
            continue
        collected_boxes.append(box_detect[idx])
        collected_scores.append(conf_detect[idx])

    if not collected_boxes:
        return np.empty((0, 4)), np.array([])

    merged_boxes = np.vstack(collected_boxes)
    merged_scores = np.concatenate(collected_scores)
    final_boxes, final_scores = _apply_nms(merged_boxes, merged_scores, conf, iou_thresh)
    return final_boxes, np.round_(final_scores, decimals=3)


# Draw
def Draw(model, img, conf=0.5, iou_thresh=0.45, imgsz=960, multiscale=True, aux_scales=None):
    boxes, conf_scores = detectImg(
        model,
        img,
        conf=conf,
        iou_thresh=iou_thresh,
        imgsz=imgsz,
        multiscale=multiscale,
        aux_scales=aux_scales,
    )
    for num, box in enumerate(boxes):
        img = cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            img,
            str(conf_scores[num]),
            (int(box[0]), int(box[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return img


# Task detect video
def detectVideo(args, model):
    '''
    Args: 
        imgsz (int): Input of size image. Default: 960
        input (str): Path of input data. Default: 337.png
        output (str): Path of output data. Default: output
        model (str): Path of model. Default: ./Model/Boat-detect-medium.pt
        conf (float): Score confidence. Default: 0.6
        iou_threshold (float): IOU threshold. Default: 0.5
        video (bool): Input is video. Default: False
        detect (bool): Task is detection. Default: False
        tracking (bool): Task is tracking. Default: False
        track_buffer (int): buffer to calculate the time when to remove tracks. Default: 30
        match_thresh (float): Matching threshold for tracking in bytetrack. Default: 0.5
        time_check_state (float): Time to update state of ship (second). Default: 1.5
        
    '''
    video = cv2.VideoCapture(args.input)
    if video.isOpened() is False:
        print("Error reading video file")
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    print(size)
    result = cv2.VideoWriter(
        "./Data/Output/" + args.output + ".avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        size,
    )
    while True:
        ret, frame = video.read()
        if ret:
            frame = Draw(
                model,
                frame,
                conf=args.conf,
                iou_thresh=args.iou_threshold,
                imgsz=args.imgsz,
                multiscale=args.multiscale,
                aux_scales=args.aux_imgsz,
            )
            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    video.release()
    result.release()
    cv2.destroyAllWindows()
    print("The video was successfully detected")
    print("The video was successfully saved")
