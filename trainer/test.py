import numpy as np




def main():
    dist = np.array([[2,3,1],[4,5,1],[1,2,3]])
    d_shape = dist.shape
    print(d_shape)
    A = np.zeros(shape=(15,15))
    batch_size=5

    for i in range(batch_size):
        for x in range(d_shape[0]):
            for y in range(d_shape[1]):
                A[i*d_shape[0]+x][i*d_shape[1]+y] = dist[x][y]


    print(A)


if __name__=='__main__':
    main()