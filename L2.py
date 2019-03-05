import numpy as np
from SubPixelAlignment import SubPixelAlignment


class L2(SubPixelAlignment):

    def __init__(self):
        super().__init__()

    def L2residualComputation(self, img_no):

        matrix2 = self.imgs[img_no]
        D2 = np.zeros_like(matrix2)

        movement_x = np.zeros((self.tile_rows * 16, self.tile_cols * 16))
        movement_y = np.zeros((self.tile_rows * 16, self.tile_cols * 16))

        for i in range(self.tile_rows):
            for j in range(self.tile_cols):

                template_norm = np.linalg.norm(self.tilevec[i, j, :, :], ord=1)

                min_D2 = 10000

                for k in range(-4, 5):
                    for l in range(-4, 5):

                        if self.isSafe(i, j, k, l):

                            seq_image_norm = np.linalg.norm(matrix2[(i*16 + k):(i*16 + k + 16), (j*16 + l):(j*16 + l + 16)]
                                                            , ord=1)

                            correlation = np.sum(self.tilevec[i, j, :, :]*matrix2[(i*16+k):(i*16+k+16), (j*16+l):(j*16+l+16)])

                            current_D2 = template_norm + seq_image_norm - 2*correlation

                            if min_D2 > current_D2:
                                min_D2 = current_D2
                                min_k = k
                                min_l = l

                movement_x[i * 16:i * 16 + 16, j * 16:j * 16 + 16] = min_k
                movement_y[i * 16:i * 16 + 16, j * 16:j * 16 + 16] = min_l

        return movement_x, movement_y

    def l2_distance_all(self):

        self.tilevec = self.make_tiles(self.ref_img)
        self.tile_rows = self.tilevec.shape[0]
        self.tile_cols = self.tilevec.shape[1]

        dir1_sequence = np.zeros((len(self.imgs), self.tile_rows * 16, self.tile_cols * 16))
        dir2_sequence = np.zeros((len(self.imgs), self.tile_rows * 16, self.tile_cols * 16))

        for i in range(len(self.imgs)):
            if i != self.ref_img_arg:
                dir1_sequence[i], dir2_sequence[i] = self.L2residualComputation(i)

        return dir1_sequence, dir2_sequence





