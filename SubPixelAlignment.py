import numpy as np
import glob
import cv2


class SubPixelAlignment:

    def __init__(self):
        self.imgs = []
        self.ref_img = []
        self.ref_img_arg = np.inf
        self.tilevec = []

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


def main():

    sp_object = SubPixelAlignment()
    img_vec = sp_object.get_image_vector()
    sp_object.find_reference()
    tilevec = sp_object.make_tiles(sp_object.ref_img)

    print("Aligning...")


if __name__ == "__main__":
    main()
