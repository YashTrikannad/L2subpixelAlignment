import numpy as np
from SubPixelAlignment import SubPixelAlignment


class L1(SubPixelAlignment):

    def __init__(self):
        super().__init__()

    def l1_distance(self, img_no):

        matrix2 = self.imgs[img_no]

        movement_1 = np.zeros((self.tile_rows * 16, self.tile_cols * 16))
        movement_2 = np.zeros((self.tile_rows * 16, self.tile_cols * 16))

        for i in range(self.tile_rows):
            for j in range(self.tile_cols):
                norm = np.linalg.norm(self.tilevec[i, j, :, :])
                min_norm = 10000
                min_l = 0
                min_k = 0
                for k in range(-1, 2):
                    for l in range(-1, 2):

                        if self.isSafe(i, j, k, l):

                            if k == 0 and l == 0:
                                continue
                            x = np.abs(np.linalg.norm(matrix2[(i*16+k):(i*16+k+16), (j*16+l):(j*16+l+16)]) - norm)
                            if x < min_norm:
                                min_norm = x
                                min_l = l
                                min_k = k
                movement_1[i*16:i*16+16, j*16:j*16+16] = min_l
                movement_2[i*16:i*16+16, j*16:j*16+16] = min_k

        return movement_1, movement_2

    def l1_distance_all(self):
        self.tilevec = self.make_tiles(self.ref_img)
        self.tile_rows = self.tilevec.shape[0]
        self.tile_cols = self.tilevec.shape[1]
        dir1_sequence = np.zeros((len(self.imgs), self.tile_rows*16, self.tile_cols*16))
        dir2_sequence = np.zeros((len(self.imgs), self.tile_rows*16, self.tile_cols*16))
        for i in range(len(self.imgs)):
            if i != self.ref_img_arg:
                dir1_sequence[i], dir2_sequence[i] = self.l1_distance(i)
        return dir1_sequence, dir2_sequence
