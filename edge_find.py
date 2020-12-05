import os
import glob
import numpy as np
import cv2
import tqdm

def main():
    root = "/root/volume/Cityscapes/"

    gtFine = os.path.join(root, 'gtFine')
    edge = os.path.join(root, 'edge_label')

    os.makedirs(edge, exist_ok=True)

    gt_colors = glob.glob(os.path.join(gtFine, "**/*gtFine_color.png"), recursive=True)
    gt_ids = glob.glob(os.path.join(gtFine, "**/*gtFine_labelIds.png"), recursive=True)


    for gt, id in zip(gt_colors, gt_ids):
        if 'test' in gt:
            continue
        new_name = os.path.join(edge, id[len(root):])
        new_dir = os.path.split(new_name)[0]
        os.makedirs(new_dir, exist_ok=True)

        edge_img = get_edge_img(gt)
 
        id_img = cv2.imread(id, cv2.IMREAD_GRAYSCALE)

        color_thresholds = (edge_img[:,:] == 255)

        id_img[color_thresholds] = [255]

        cv2.imwrite(new_name, id_img)

        # break


def get_edge_img(gt_img_path):

    # gt_img = cv2.imread(gt_img, cv2.IMREAD_GRAYSCALE)
    gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

    # cv2.imshow('image', gt_img)
    canny_img = cv2.Canny(gt_img, 0.1, 40)
    # edge_img = cv2.Sobel(cv2.CV_8U)
    # edge_img = gt_img

    # edge_line = np.where(canny_img == 255)

    edge_img = canny_img.copy()

    edge_img[1:, :] = np.bitwise_or(edge_img[1:, :], canny_img[:-1, :])
    edge_img[2:, :] = np.bitwise_or(edge_img[2:, :], canny_img[:-2, :])
    edge_img[3:, :] = np.bitwise_or(edge_img[3:, :], canny_img[:-3, :])
    edge_img[:-1, :] = np.bitwise_or(edge_img[:-1, :], canny_img[1:, :])
    edge_img[:-2, :] = np.bitwise_or(edge_img[:-2, :], canny_img[2:, :])
    edge_img[:-3, :] = np.bitwise_or(edge_img[:-3, :], canny_img[3:, :])

    edge_img[:, 1:] = np.bitwise_or(edge_img[:, 1:], canny_img[:, :-1])
    edge_img[:, 2:] = np.bitwise_or(edge_img[:, 2:], canny_img[:, :-2])
    edge_img[:, 3:] = np.bitwise_or(edge_img[:, 3:], canny_img[:, :-3])
    edge_img[:, :-1] = np.bitwise_or(edge_img[:, :-1], canny_img[:, 1:])
    edge_img[:, :-2] = np.bitwise_or(edge_img[:, :-2], canny_img[:, 2:])
    edge_img[:, :-3] = np.bitwise_or(edge_img[:, :-3], canny_img[:, 3:])

    #edge_img = np.stack([edge_img, edge_img, edge_img], axis=2)

    #color_thresholds = (edge_img[:,:,0] == 255) & (edge_img[:,:,1] == 255) & (edge_img[:,:,2] == 255)

    #gt_img[color_thresholds] = [255,255,255]

    return edge_img


if __name__ == "__main__":
    main()
