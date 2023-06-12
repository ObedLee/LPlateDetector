from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import glob
import time
import unicodedata

# 실행방법
# python detect.py --dataset dataset --detectset result --detect detect.dat
# python accuracy.py -r ref.dat -d detect.dat

MIN_AREA = 100
MIN_WIDTH, MIN_HEIGHT = 6, 30
MIN_RATIO, MAX_RATIO = 0.25, 1.0

PLATE_WIDTH_PADDING = 1.3
PLATE_HEIGHT_PADDING = 2.0

MAX_DIAG_MULTIPLIER = 5
MAX_ANGLE_DIFF = 7
MAX_AREA_DIFF = 0.2
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3

def find_chars(contour_list):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']])
                                      - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLIER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        try:
            unmatched_contour = np.take(contour_list, unmatched_contour_idx)
            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
            break
        except IndexError:
            break

    return matched_result_idx


def detect_car_license_plate(img, fname, ksize):
    # 개별 작성
    gray = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (63, 49))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    gray = cv2.add(gray, tophat)
    gray = cv2.subtract(gray, blackhat)

    blur = cv2.GaussianBlur(gray, ksize=(ksize, ksize), sigmaX=5)

    thresh = cv2.adaptiveThreshold(
        blur,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=9
    )

    _contours, _ = cv2.findContours(
        thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours = []
    for contour in _contours:
        x, y, w, h = cv2.boundingRect(contour)

        contours.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    possible_contours = []

    cnt = 0
    for d in contours:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    plate_infos = []
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    if plate_infos:
        info = plate_infos[-1]
        for i in plate_infos:
            x, y, w, h = i.values()
            if x < info['x']:
                info['x'] = x
            if y < info['y']:
                info['y'] = y
            if w > info['w']:
                info['w'] = w
            if h > info['h']:
                info['h'] = h
        if info['x'] < 0:
            info['x'] = 0
        if info['y'] < 0:
            info['y'] = 0

    else:
        if ksize > 2:
            return detect_car_license_plate(img, fname, ksize - 2)
        else:
            info = {'x': 120, 'y': 230, 'w': 830, 'h': 470}

    points = np.concatenate([[info['x'], info['y']],
                             [info['x']+info['w'], info['y']+info['h']]])

    return points


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
    ap.add_argument("-r", "--detectset", required = True, help = "path to the detectset folder")
    ap.add_argument("-f", "--detect", required = True, help = "path to the detect file")
    args = vars(ap.parse_args())
    
    dataset = args["dataset"]
    detectset = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if not os.path.isdir(detectset):
        os.mkdir(detectset)

    # 결과 영상 표시 여부
    verbose = False

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩

    # 자동차 번호판 파일 리스트
    filelist = glob.glob(dataset + "/*.jpg") + glob.glob(dataset + "/*.JPG") + glob.glob(dataset + "/*.jpeg")+glob.glob(dataset + "/*.JPEG")

    start = time.time()  # 시작 시간 저장
    for imagePath in filelist:
        imagePath = unicodedata.normalize('NFC', imagePath)
        print(imagePath, '처리중...')

        # 영상을 불러오고 그레이 스케일 영상으로 변환
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 자동차 번호판 검출
        points = detect_car_license_plate(gray, imagePath, 9)

        # 자동차 번호판 영역 표시
        detectimg = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 2)  # 이미지에 사각형 그리기

        # 결과 영상 저장
        loc1 = imagePath.rfind("/")
        loc2 = imagePath.rfind(".")
        fname = 'result/' + imagePath[loc1 + 1: loc2] + '_res.jpg'
        cv2.imwrite(fname, detectimg)

        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(points[0]))
        f.write("\t")
        f.write(str(points[1]))
        f.write("\t")
        f.write(str(points[2]))
        f.write("\t")
        f.write(str(points[3]))
        f.write("\n")

        if verbose:
            cv2.imshow("image", image)
            cv2.waitKey(0)

    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간