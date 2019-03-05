import numpy as np


dist_mtx=np.zeros((3,3))
for k in range(-1, 2):
    for l in range(-1, 2):
        if self.isSafe(i, j, k, l):

            if k == 0 and l == 0:
                continue

            x = np.abs(np.linalg.norm(matrix2[(i*16+(min_k+k)):(i*16+(min_k+k+16)), (j*16+(l+min_l)):(j*16+min_l+l+16)]) - norm)
            dist_mtx[k+1,l+1]=x

fa_11=np.array([[1,-2,1],[2,-4,2],[1,-2,1]])/4
fa_12=np.array([[1,0,-1],[0,0,0],[-1,0,1]])/4
fa_21=fa_12
fa_22=np.array([[1,0,-1],[0,0,0],[-1,0,1]])/4
fb_1=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/4
fb_2=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])/4
fc=np.array([[-1,2,-1],[2,12,2],[-1,2,-1]])/4

A=np.array((2,2))
A[0,0]=np.sum(fa_11*dist_mtx)
A[0,0]=np.max((0,A[0,0]))
A[0,1]=np.sum(fa_12*dist_mtx)
A[1,0]=A[0,1]
A[1,1]=np.sum(fa_22*dist_mtx)
A[1,1]=np.max((0,A[1,1]))
if(np.linalg.det(A)<0):
    A[1,0]=0
    A[0,1]=0


b=np.array((2,1))
b[0,0]=np.sum(fb_1*dist_mtx)
b[1,0]=np.sum(fb_2*dist_mtx)

c=np.sum(fc*dist_mtx)
mu=np.matmul(np.linalg.inv(-a),b)
mu[0,0]+=min_l
mu[1,0]+=min_k
movement_1[i*16:i*16+16, j*16:j*16+16] = mu[0,0]
movement_2[i*16:i*16+16, j*16:j*16+16] = mu[1,0]
