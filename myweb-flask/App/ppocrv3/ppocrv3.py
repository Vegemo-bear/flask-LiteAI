import copy

import numpy as np
import cv2
import onnxruntime
from shapely.geometry import Polygon
import pyclipper
import math


def pp_img_preprocess(img):
    img = np.frombuffer(img, dtype='uint8')
    img = cv2.imdecode(img, 1)
    src_h, src_w, _ = img.shape
    resize_h, resize_w = [736, 1280]
    ori_h, ori_w = img.shape[:2]
    ratio_h = float(resize_h) / ori_h
    ratio_w = float(resize_w) / ori_w
    img = cv2.resize(img, (int(resize_w), int(resize_h)))
    scale = np.float32(1.0 / 255.0)
    mean = [0.485, 0.456, 0.406]
    mean = np.array(mean).reshape((1, 1, 3)).astype('float32')
    std = [0.229, 0.224, 0.225]
    std = np.array(std).reshape(1, 1, 3).astype('float32')
    img = (img.astype('float32') * scale - mean) / std
    pre_img = img.transpose((2, 0, 1))[None]
    return pre_img, ratio_h, ratio_w, src_h, src_w


def pp_refer(img):
    session = onnxruntime.InferenceSession("App/onnxfile/ppocrv3/ppocrv3_det.onnx", providers=["CPUExecutionProvider"])
    prob = session.run(["sigmoid_0.tmp_0"], {"x": img})[0]
    return prob


def pp_postprocess(prob, ratio_h, ratio_w, src_h, src_w):
    thresh = 0.3
    min_size = 3
    max_candidates = 1000
    unclip_ratio = 2
    box_thresh = 0.6
    score_mode = "fast"

    pred = prob[:, 0, :, :]
    segmentation = pred > thresh
    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = src_h, src_w, ratio_h, ratio_w
        mask = segmentation[batch_index]
        outs = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = outs[0], outs[1]
        num_contours = min(len(contours), max_candidates)
        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            bounding_box = cv2.minAreaRect(contour)
            points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if points[1][1] > points[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if points[3][1] > points[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2
            box = [
                points[index_1], points[index_2], points[index_3], points[index_4]
            ]
            points = box
            sside = min(bounding_box[1])
            if sside < min_size:
                continue
            points = np.array(points)
            score = 0
            if score_mode == "fast":
                bitmap = pred
                _box = points.reshape(-1, 2)
                h, w = bitmap[0].shape[:2]
                box = _box.copy()
                xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
                xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
                ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
                ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)
                mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
                box[:, 0] = box[:, 0] - xmin
                box[:, 1] = box[:, 1] - ymin
                cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
                score = cv2.mean(bitmap[0][ymin:ymax + 1, xmin:xmax + 1], mask)[0]
            if box_thresh > score:
                continue

            poly = Polygon(points)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = np.array(offset.Execute(distance))
            box = expanded.reshape(-1, 1, 2)
            bounding_box = cv2.minAreaRect(box)
            points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if points[1][1] > points[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if points[3][1] > points[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2

            box = [
                points[index_1], points[index_2], points[index_3], points[index_4]
            ]
            box, sside = box, min(bounding_box[1])
            if sside < min_size + 2:
                continue
            box = np.array(box)
            height, width = segmentation[0].shape
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * src_w), 0, src_w)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * src_h), 0, src_h)
            boxes.append(box.astype("float32"))
            scores.append(score)
        boxes = np.array(boxes, dtype='float32')
        scores = scores
        boxes_batch.append({'points': boxes})

    post_result = boxes_batch
    return post_result


def pp_rec(img_list):
    img_num = len(img_list)
    width_list = []
    for img in img_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    indices = np.argsort(np.array(width_list))
    rec_res = [['', 0.0]] * img_num
    batch_num = 8
    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        imgC, imgH, imgW = (3, 48, 320)
        max_wh_ratio = imgW / imgH
        for ino in range(beg_img_no, end_img_no):
            h, w = img_list[indices[0]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            imgC, imgH, imgW = (3, 48, 320)
            imgW = 320
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
            resized_image = resized_image.astype('float32')
            resized_image = resized_image.transpose((2, 0, 1)) / 255
            resized_image -= 0.5
            resized_image /= 0.5
            padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
            padding_im[:, :, 0:resized_w] = resized_image
            norm_img = padding_im
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()

    session = onnxruntime.InferenceSession("App/onnxfile/ppocrv3/ppocrv3_rec.onnx", providers=["CPUExecutionProvider"])
    prob = session.run(["softmax_5.tmp_0"], {"x": norm_img_batch})[0]

    preds = prob
    preds_idx = prob.argmax(axis=2)
    preds_prob = preds.max(axis=2)

    selection = np.ones(len(preds_idx[0]), dtype=bool)
    selection[1:] = preds_idx[0][1:] != preds_idx[0][:-1]
    selection &= preds_idx[0] != 0

    character_str = ["blank"]
    character_dict_path = 'App/dict/ppocr_keys_v1.txt'
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character_str.append(line)
    character_str.append(" ")

    char_list = [
        character_str[text_id]
        for text_id in preds_idx[0][selection]
    ]

    conf_list = []
    if preds_prob is not None:
        conf_list = preds_prob[0][selection]
    if len(conf_list) == 0:
        conf_list = [0]

    text = ''.join(char_list)
    result_list = []
    result_list.append(text)
    return result_list


def pp_generate_result(src_img, post_result, save_det_path):
    rec_res = []
    img_crop_list = []
    src_img = cv2.imread(src_img)
    dt_boxes = post_result[0]['points']
    if len(dt_boxes) > 0:
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)
        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        dt_boxes = _boxes
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])

            box = np.array(tmp_box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_img, [box], True, color=(255, 255, 0), thickness=1)

            img_crop_width = int(
                max(
                    np.linalg.norm(tmp_box[0] - tmp_box[1]),
                    np.linalg.norm(tmp_box[2] - tmp_box[3])))
            img_crop_height = int(
                max(
                    np.linalg.norm(tmp_box[0] - tmp_box[3]),
                    np.linalg.norm(tmp_box[1] - tmp_box[2])))
            pts_std = np.float32([[0, 0], [img_crop_width, 0],
                                  [img_crop_width, img_crop_height],
                                  [0, img_crop_height]])
            M = cv2.getPerspectiveTransform(tmp_box, pts_std)
            dst_img = cv2.warpPerspective(
                src_img,
                M, (img_crop_width, img_crop_height),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC)
            dst_img_height, dst_img_width = dst_img.shape[0:2]
            if dst_img_height * 1.0 / dst_img_width >= 1.5:
                dst_img = np.rot90(dst_img)
            img_crop = dst_img
            img_crop_list.append(img_crop)
            rec_res.append(pp_rec(img_crop_list))

        cv2.imwrite(save_det_path, src_img)
        return rec_res

