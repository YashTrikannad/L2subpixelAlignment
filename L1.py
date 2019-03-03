import numpy as np
from SubPixelAlignment import SubPixelAlignment


class L1(SubPixelAlignment):

    def __init__(self):
        super().__init__()

    def l1_distance(self, img_no):
        self.tilevec = self.make_tiles(self.ref_img)
        matrix2 = self.imgs[img_no]

        tile_rows = self.tilevec[0]
        tile_cols = self.tilevec[1]

        movement_dir1 = np.zeros_like((tile_rows, tile_cols))
        movement_dir2 = np.zeros_like((tile_rows, tile_cols))

        for i in range(tile_rows):
            for j in range(tile_cols):
                norm = np.linalg.norm(self.tilevec[i, j, :, :])
                min_norm = 10000
                min_l = 0
                min_k = 0
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if k == 0 and l == 0:
                            continue
                        x = np.linalg.norm(matrix2[(i*16+k):(i*16+k+16), (j*16+l):(j*16+l+16)]) - norm
                        if x < min_norm:
                            min_norm = x
                            min_l = l
                            min_k = k
                movement_dir1[i, j] = min_l
                movement_dir2[i, j] = min_k

        return movement_dir1, movement_dir2

    def l1_distance_all(self):
        dir1_sequence = np.empty((1, len(self.imgs)), dtype=object)
        dir2_sequence = np.empty((1, len(self.imgs)), dtype=object)
        for i in range(len(self.imgs)):
            if i != self.ref_img_arg:
                dir1_sequence[i], dir2_sequence[i] = self.l1_distance(i)

