import cv2
from skimage import segmentation

mark_boundaries = segmentation.mark_boundaries


def press_wasdqe_to_adjust_parameter_of_felz(img):
    paras = [64, 0.5, 128]  # pieces
    # paras = [1, 0.8, 20]  # default

    act_dict = {0: (0, 1.0)}
    act_dict.update(zip([119, 97, 113], [(i, 1.2) for i in range(len(paras))], ))  # _KeyBoard: W A Q
    act_dict.update(zip([115, 100, 101], [(i, 0.8) for i in range(len(paras))], ))  # KeyBoard: S D E

    key = 0
    while True:
        if key != -1:
            i, multi = act_dict[key]
            paras[i] *= multi
            print(key, paras)

            seg_map = segmentation.felzenszwalb(img,
                                                scale=int(paras[0]),
                                                sigma=paras[1],
                                                min_size=int(paras[2]))
            show = mark_boundaries(img, seg_map)
            cv2.imshow('', show)

            wait_time = 1
        else:
            wait_time = 100

        key = cv2.waitKey(wait_time)
        break
    cv2.imwrite('tiger_felz.jpg', show * 255)


def press_wasdqe_to_adjust_parameter_of_slic(img):
    paras = [100, 10000, 10]  # pieces
    # paras = [100, 10, 10]  # default
    # paras = [16, 64, 6]  # appropriate

    act_dict = {0: (0, 1.0)}
    act_dict.update(zip([119, 97, 113], [(i, 1.2) for i in range(len(paras))], ))  # _KeyBoard: W A Q
    act_dict.update(zip([115, 100, 101], [(i, 0.8) for i in range(len(paras))], ))  # KeyBoard: S D E

    key = 0
    while True:
        if key != -1:
            i, multi = act_dict[key]
            paras[i] *= multi
            print(key, paras)

            seg_map = segmentation.slic(img,
                                        compactness=int(paras[0]),
                                        n_segments=int(paras[1]),
                                        max_iter=int(paras[2]), )
            show = mark_boundaries(img, seg_map)
            cv2.imshow('', show)

            wait_time = 1
        else:
            wait_time = 100

        key = cv2.waitKey(wait_time)
    #     break
    # cv2.imwrite('tiger_slic.jpg', show * 255)


if __name__ == '__main__':
    image = cv2.imread('image/tiger.jpg')
    press_wasdqe_to_adjust_parameter_of_felz(image)
    # press_wasdqe_to_adjust_parameter_of_slic(image)
