import glob
import cv2
from L1 import *


class SubPixelAlignment:

    def __init__(self):
        self.imgs = []
        self.ref_img = []
        self.ref_img_arg = np.inf
        self.tilevec = []
        self.tile_rows = 0
        self.tile_cols = 0

    def get_image_vector(self):
        self.imgs = [cv2.imread(file, 0) for file in glob.glob("/home/yash/PycharmProjects/L2subpixelAlignment/Test1/*.jpg")]
        return self.imgs

    def make_tiles(self, image):
        image = np.asarray(image)
        tile_nrows = int(image.shape[0]/16)
        tile_ncols = int(image.shape[1]/16)
        self.tilevec = np.zeros((tile_nrows, tile_ncols, 16, 16))
        for i in range(tile_nrows):
            for j in range(tile_ncols):
                self.tilevec[i, j, :, :] = image[i:i+16, j:j+16]
        return self.tilevec

    def find_reference(self):
        min_fm = np.inf
        min_fm_arg = -1
        for i in range(len(self.imgs)):
            fm = cv2.Laplacian(self.imgs[i], cv2.CV_64F).var()
            if fm < min_fm:
                min_fm = fm
                min_fm_arg = i
        self.ref_img = self.imgs[i]
        self.ref_img_arg = min_fm_arg

    def isSafe(self, i, j, k, l):
        safety1 = i + k
        safety2 = j + l

        if safety1 < 0 or safety1 > self.tile_rows or safety2 < 0 or safety2 > self.tile_cols:
            return False
        else:
            return True


def main():

    sp_object = L1()
    sp_object.get_image_vector()
    sp_object.find_reference()
    sequence_dir1, sequence_dir2 = sp_object.l1_distance_all()

    print("Aligning...")


if __name__ == "__main__":
    main()
