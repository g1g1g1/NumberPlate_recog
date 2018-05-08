'''
Code source: https://github.com/matthewearl/deep-anpr
'''

#encoding=utf-8
__all__ = (
    'detect',
    'post_process',
)

import time
import collections
import math
import cv2
import numpy
import tensorflow as tf
import common
import model

def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))

def detect(im, param_vals):
    """
    Detect number plates in an image.
    :param im:
        Image to detect number plates in.
    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.
    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.
    """

    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.

    # 리턴 인자 kp 추가
    x, y, params, kp = model.get_detect_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            feed_dict.update({kp: 1.0})                        # 실제 사용시에는 keep_prob=1.0으로 설정
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups 
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


def post_process(matches):
    """
    Take an iterable of matches as returned by `detect` and merge duplicates.
    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.
    """
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        present_probs = numpy.array([m[2] for m in group_matches])
        letter_probs = numpy.stack(m[3] for m in group_matches)

        yield letter_probs[numpy.argmax(present_probs)] #실제 쓰이는 인자만 리턴!


def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


if __name__ == "__main__":
    print("데이터셋 validating 시작")
    f = numpy.load("weights0417ver_mini_train.npz")   # weight 파일
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    for idx in range(1, 99):
        target_img = "G:/ewha_project/test_data/in"+str(idx)+".jpg"
        print("detect.py 실행중입니다. 대상 이미지:", target_img)
        im2 = cv2.imread(target_img)  # input image
        print("with resizing input image >>>>")
        # resizing - 비율 기준으로
        # 정면, 정면-위쪽에서 찍힌 경우 가로 480px 필요 - in30.jpg, in 31.jpg
        # 측면으로 약간(좌측) 치우친 경우 500px 필요 - in32.jpg
        # 측면으로 심하게(우측) 치우친 경우 600px 필요 - in33.jpg
    
        if(im2.shape[1]>600):
            r = 600. / im2.shape[1]
            dsize = (600, int(im2.shape[0]*r))
            im2 = cv2.resize(im2, dsize)

        im_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) / 255.
        start_time = time.time()

        for letter_probs in post_process(detect(im_gray, param_vals)):
            code = letter_probs_to_code(letter_probs)

            if code is "":
                print("예측이 제대로 이루어지지 않음.")
            else:
                print("작업 결과물:", code)

        end_time = time.time()
        print("detect에 걸린 시간:", end_time-start_time)
        print("==================================================")

    
