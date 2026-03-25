#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# all this collection u can install with requirements.txt -- pip install -r requirements.txt
from collections import defaultdict, deque
from pathlib import Path
import time
import cv2
import torch
from ultralytics import YOLO

VIDEO_PATH = "size.mp4" # u need to put your mp4 video name here (and it needs to be in this same repository, of course)
MODEL_PATH = "yolo26l.pt"
OUTPUT_PATH = f"SAIDA - CONTAGEM - {VIDEO_PATH}"

VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
UNIFIED_LABEL = "vehicle"

IMGSZ = 640
CONF = 0.35
TRACK_HISTORY = 20
VID_STRIDE = 1
SHOW_WINDOW = True
TRACKER_CFG = "bytetrack.yaml"

# =========================
# CONFIGURAÇÃO DA LINHA
# =========================
LINE_MODE = "horizontal"  # "horizontal" ou "vertical" # if u want an horizontal line, let it like it is. if u want a vertical, just write "vertical" instead "horizontal"

# Horizontal
LINE_X1 = 100
LINE_X2 = 1100
LINE_Y = 500

# Vertical
LINE_X = 600
LINE_Y1 = 100
LINE_Y2 = 700

CROSS_TOLERANCE = 2


def get_device():
    """
    Usa GPU NVIDIA via CUDA se disponível.
    Caso contrário, cai para CPU.
    """
    if torch.cuda.is_available():
        print("[INFO] CUDA disponível: True")
        print(f"[INFO] GPU detectada: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("[INFO] CUDA disponível: False")
        print("[INFO] Usando CPU")
        return "cpu"


def create_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Não foi possível criar o vídeo: {output_path}")
    return writer


def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def crossed_line(prev, curr):
    """
    Detecta cruzamento da linha configurada.
    Funciona para modo horizontal ou vertical.
    """
    if prev is None or curr is None:
        return False

    px, py = prev
    cx, cy = curr

    if LINE_MODE == "horizontal":
        if not (LINE_X1 <= cx <= LINE_X2):
            return False

        crossed_down = py < (LINE_Y - CROSS_TOLERANCE) and cy >= (LINE_Y - CROSS_TOLERANCE)
        crossed_up = py > (LINE_Y + CROSS_TOLERANCE) and cy <= (LINE_Y + CROSS_TOLERANCE)
        return crossed_down or crossed_up

    elif LINE_MODE == "vertical":
        if not (LINE_Y1 <= cy <= LINE_Y2):
            return False

        crossed_right = px < (LINE_X - CROSS_TOLERANCE) and cx >= (LINE_X - CROSS_TOLERANCE)
        crossed_left = px > (LINE_X + CROSS_TOLERANCE) and cx <= (LINE_X + CROSS_TOLERANCE)
        return crossed_right or crossed_left

    return False


def draw_line(frame):
    if LINE_MODE == "horizontal":
        cv2.line(frame, (LINE_X1, LINE_Y), (LINE_X2, LINE_Y), (0, 0, 255), 3)
        cv2.putText(
            frame,
            "Linha horizontal",
            (LINE_X1, max(30, LINE_Y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
    else:
        cv2.line(frame, (LINE_X, LINE_Y1), (LINE_X, LINE_Y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            "Linha vertical",
            (LINE_X + 10, max(30, LINE_Y1 + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


def draw_box(frame, box, track_id, conf):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

    if track_id == -1:
        text = f"{UNIFIED_LABEL} sem_id {conf:.2f}"
    else:
        text = f"{UNIFIED_LABEL} ID:{track_id} {conf:.2f}"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    y_top = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y1), (40, 220, 40), -1)

    cv2.putText(
        frame,
        text,
        (x1 + 4, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def draw_track(frame, pts):
    pts = list(pts)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = max(1, int(5 - i * 0.15))
        cv2.line(frame, pts[i - 1], pts[i], (0, 180, 255), thickness)


def main():
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {VIDEO_PATH}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    device = get_device()
    use_half = device != "cpu"

    print(f"[INFO] Dispositivo selecionado: {device}")
    print(f"[INFO] Half precision: {use_half}")
    print(f"[INFO] Carregando modelo: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_writer(OUTPUT_PATH, fps, width, height)

    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY))
    last_centers = {}
    counted_ids = set()
    total_count = 0

    frame_count = 0
    processed_count = 0
    t0 = time.time()

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
        frame_count += 1
        frame = result.orig_img.copy()

        draw_line(frame)

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)

            for box, conf, track_id in zip(xyxy, confs, ids):
                cx, cy = center_of_box(box)

                cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)

                if track_id != -1:
                    track_history[track_id].append((cx, cy))
                    draw_track(frame, track_history[track_id])

                    prev = last_centers.get(track_id)

                    if track_id not in counted_ids and crossed_line(prev, (cx, cy)):
                        total_count += 1
                        counted_ids.add(track_id)

                    last_centers[track_id] = (cx, cy)

                draw_box(frame, box, track_id, conf)

        processed_count += 1
        elapsed = time.time() - t0
        fps_proc = processed_count / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            frame,
            f"Contagem: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"FPS proc.: {fps_proc:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("Contagem de Veiculos", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - t0
    print(f"[OK] Saída salva em: {OUTPUT_PATH}")
    print(f"[OK] Total contado: {total_count}")
    print(f"[OK] Tempo total: {total_time:.2f}s")


if __name__ == "__main__":
    main()
