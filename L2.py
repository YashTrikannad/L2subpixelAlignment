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

                template_norm = np.linalg.norm(self.tilevec[i, j, :, :], ord=2)

                min_D2 = 10000

                for k in range(-4, 5):
                    for l in range(-4, 5):

                        if self.isSafe(i, j, k, l):

                            seq_image_norm = np.linalg.norm(matrix2[(i*16 + k):(i*16 + k + 16), (j*16 + l):(j*16 + l + 16)]
                                                            , ord=2)

                            correlation = np.sum(self.tilevec[i, j, :, :]*matrix2[(i*16+k):(i*16+k+16), (j*16+l):(j*16+l+16)])

                            current_D2 = template_norm + seq_image_norm - 2*correlation

                            if min_D2 > current_D2:
                                min_D2 = current_D2
                                min_k = k
                                min_l = l

                movement_x[i * 16:i * 16 + 16, j * 16:j * 16 + 16] = min_k
                movement_y[i * 16:i * 16 + 16, j * 16:j * 16 + 16] = min_l

                dist_mtx = np.zeros((3, 3))
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if self.isSafe(i, j, k, l):
                            seq_image_norm = np.linalg.norm(
                                matrix2[(i * 16 + k):(i * 16 + k + 16), (j * 16 + l):(j * 16 + l + 16)]
                                , ord=2)

                            correlation = np.sum(self.tilevec[i, j, :, :] * matrix2[(i * 16 + k):(i * 16 + k + 16),
                                                                            (j * 16 + l):(j * 16 + l + 16)])

                            current_D2 = template_norm + seq_image_norm - 2 * correlation
                            dist_mtx[k + 1, l + 1] = current_D2

                fa_11 = np.array([[1, -2, 1], [2, -4, 2], [1, -2, 1]]) / 4
                fa_12 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
                fa_21 = fa_12
                fa_22 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
                fb_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
                fb_2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4
                fc = np.array([[-1, 2, -1], [2, 12, 2], [-1, 2, -1]]) / 4

                A = np.zeros((2, 2))
                A[0, 0] = np.sum(fa_11 * dist_mtx)
                A[0, 0] = np.max((0, A[0, 0]))
                A[0, 1] = np.sum(fa_12 * dist_mtx)
                A[1, 0] = A[0, 1]
                A[1, 1] = np.sum(fa_22 * dist_mtx)
                A[1, 1] = np.max((0, A[1, 1]))

                if (np.linalg.det(A) < 0):
                    A[1, 0] = 0
                    A[0, 1] = 0

                b = np.zeros((2, 1))
                b[0, 0] = np.sum(fb_1 * dist_mtx)
                b[1, 0] = np.sum(fb_2 * dist_mtx)

                c = np.sum(fc * dist_mtx)
                mu = np.matmul(np.linalg.inv(-A), b)
                mu[0, 0] += min_l
                mu[1, 0] += min_k

                movement_x[i * 16:i * 16 + 16, j * 16:j * 16 + 16] = mu[0, 0]
                movement_y[i * 16:i * 16 + 16, j * 16:j * 16 + 16] = mu[1, 0]

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





