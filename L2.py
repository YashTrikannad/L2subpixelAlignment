import numpy as np
from SubPixelAlignment import SubPixelAlignment


class L2(SubPixelAlignment):

    def __init__(self):
        super().__init__()

    # def L2residualComputation(self, img_no):
    #
    #     matrix2 = self.imgs[img_no]
    #
    #     self.tilevec = self.make_tiles(self.ref_img)
    #     self.tile_rows = self.tilevec.shape[0]
    #     self.tile_cols = self.tilevec.shape[1]
    #
    #     for i in range(self.tile_rows):
    #         for j in range(self.tile_cols):
    #
    #             template_norm = np.linalg.norm(self.tilevec[i, j, :, :])
    #
    #             for k in range(-4, 5):
    #                 for l in range(-4, 5):
    #
    #                     np.linalg.norm(matrix2[(i*16 + k):(i*16 + k + 16), (j*16 + l):(j*16 + l + 16)]


