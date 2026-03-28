#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from pathlib import Path
import time
import cv2
import torch
from ultralytics import YOLO

# FILE_VIDEO = "terceiro.mp4" # inserir aqui o nome do video mp4 dentro de entradas



VIDEO_PATH = "contrário.mp4"
#  VIDEO_PATH = f"./entradas/{FILE_VIDEO}"
MODEL_PATH = "yolo26l.pt"
OUTPUT_PATH = f"SAIDA {VIDEO_PATH}.mp4"






VEHICLE_CLASS_IDS = [2, 3, 5, 7]
# Classes do coco database
# names:
#   0: person
#   1: bicycle
#   2: car
#   3: motorcycle
#   4: airplane
#   5: bus
#   6: train
#   7: truck
#   8: boat
#   9: traffic light
#   10: fire hydrant
#   11: stop sign
#   12: parking meter
#   13: bench
#   14: bird
#   15: cat
#   16: dog
#   17: horse
#   18: sheep
#   19: cow
#   20: elephant
#   21: bear
#   22: zebra
#   23: giraffe
#   24: backpack
#   25: umbrella
#   26: handbag
#   27: tie
#   28: suitcase
#   29: frisbee
#   30: skis
#   31: snowboard
#   32: sports ball
#   33: kite
#   34: baseball bat
#   35: baseball glove
#   36: skateboard
#   37: surfboard
#   38: tennis racket
#   39: bottle
#   40: wine glass
#   41: cup
#   42: fork
#   43: knife
#   44: spoon
#   45: bowl
#   46: banana
#   47: apple
#   48: sandwich
#   49: orange
#   50: broccoli
#   51: carrot
#   52: hot dog
#   53: pizza
#   54: donut
#   55: cake
#   56: chair
#   57: couch
#   58: potted plant
#   59: bed
#   60: dining table
#   61: toilet
#   62: tv
#   63: laptop
#   64: mouse
#   65: remote
#   66: keyboard
#   67: cell phone
#   68: microwave
#   69: oven
#   70: toaster
#   71: sink
#   72: refrigerator
#   73: book
#   74: clock
#   75: vase
#   76: scissors
#   77: teddy bear
#   78: hair drier
#   79: toothbrush









UNIFIED_LABEL = "vehicle"

IMGSZ = 704
CONF = 0.35
TRACK_HISTORY = 20
VID_STRIDE = 1
SHOW_WINDOW = True
TRACKER_CFG = "bytetrack.yaml"

# =========================
# CONFIGURAÇÃO DA LINHA
# =========================
LINE_MODE = "horizontal"  # "horizontal" ou "vertical"

# Horizontal
LINE_X1 = 100
LINE_X2 = 2000
LINE_Y = 630

# Vertical
LINE_X = 100
LINE_Y1 = 100
LINE_Y2 = 100

CROSS_TOLERANCE = 2


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def create_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# =========================
# FUNÇÃO UNIFICADA DE CRUZAMENTO
# =========================
def crossed_line(prev, curr):
    if prev is None or curr is None:
        return False

    px, py = prev
    cx, cy = curr

    if LINE_MODE == "horizontal":
        if not (LINE_X1 <= cx <= LINE_X2):
            return False

        down = py < (LINE_Y - CROSS_TOLERANCE) and cy >= (LINE_Y - CROSS_TOLERANCE)
        up = py > (LINE_Y + CROSS_TOLERANCE) and cy <= (LINE_Y + CROSS_TOLERANCE)

        return down or up

    elif LINE_MODE == "vertical":
        if not (LINE_Y1 <= cy <= LINE_Y2):
            return False

        right = px < (LINE_X - CROSS_TOLERANCE) and cx >= (LINE_X - CROSS_TOLERANCE)
        left = px > (LINE_X + CROSS_TOLERANCE) and cx <= (LINE_X + CROSS_TOLERANCE)

        return right or left

    return False


# =========================
# DESENHO DA LINHA
# =========================
def draw_line(frame):
    if LINE_MODE == "horizontal":
        cv2.line(frame, (LINE_X1, LINE_Y), (LINE_X2, LINE_Y), (0, 0, 255), 3)
    else:
        cv2.line(frame, (LINE_X, LINE_Y1), (LINE_X, LINE_Y2), (0, 0, 255), 3)


def draw_box(frame, box, track_id, conf):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

    text = f"{UNIFIED_LABEL} ID:{track_id} {conf:.2f}"
    cv2.putText(frame, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def draw_track(frame, pts):
    pts = list(pts)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 180, 255), 2)


def main():
    device = get_device()
    use_half = device != "cpu"

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_writer(OUTPUT_PATH, fps, width, height)

    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY))
    last_centers = {}
    counted_ids = set()
    total_count = 0

    results = model.track(
        source=VIDEO_PATH,
        stream=True,
        persist=True,
        tracker=TRACKER_CFG,
        conf=CONF,
        imgsz=IMGSZ,
        device=device,
        half=use_half,
        classes=VEHICLE_CLASS_IDS,
        vid_stride=VID_STRIDE,
        verbose=False
    )

    for result in results:
        frame = result.orig_img.copy()
        draw_line(frame)

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:



            xyxy = boxes.xyxy.cpu().numpy()

            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)

            for box, conf, track_id in zip(xyxy, confs, ids):

                cx, cy = center_of_box(box)

                if track_id != -1:
                    track_history[track_id].append((cx, cy))
                    draw_track(frame, track_history[track_id])

                    prev = last_centers.get(track_id)

                    if track_id not in counted_ids:
                        if crossed_line(prev, (cx, cy)):
                            total_count += 1
                            counted_ids.add(track_id)

                    last_centers[track_id] = (cx, cy)

                draw_box(frame, box, track_id, conf)

        cv2.putText(frame, f"Contagem: {total_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("Contagem", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Total contado: {total_count}")


if __name__ == "__main__":
    main()
