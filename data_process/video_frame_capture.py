import cv2
import os

from tqdm import tqdm
import numpy as np
import json


def extract_frames(input_video, input_video_id, k):
    cap = cv2.VideoCapture(input_video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    samples = (np.linspace(0, total_frames - 1, k)).tolist()

    samples = [int(i) for i in samples]

    output_folder = r'/home/qian/chenshangheng/graph360/topic/data/frame'

    os.makedirs(output_folder, exist_ok=True)

    index = 0
    output_path = os.path.join(output_folder, f"{input_video_id}_{index}.jpg")
    if os.path.exists(output_path):
        return
    for sample in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample)
        success, frame = cap.read()
        # print(input_video)
        if success != True:
            print(input_video)
            break
        output_path = os.path.join(output_folder, f"{input_video_id}_{index}.jpg")
        index += 1
        cv2.imwrite(output_path, frame)
    cap.release()


if __name__ == '__main__':

    source_path = r'/home/qian/chenshangheng/graph360/topic/data/available_dataset.json'

    # files = os.listdir(path)
    # files = [f for f in files if not f.startswith('.')]

    k = 8

    dataset = json.load(open(source_path, encoding='utf8'))
    for vid, data in tqdm(dataset.items()):
        input_video_path = dataset[vid]['视频地址']
        extract_frames(input_video_path, vid, k)