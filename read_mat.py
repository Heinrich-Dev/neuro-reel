import scipy.io

file = "pvc-1/crcns-ringach-data/neurodata/ac1/ac1_u004_000.mat"

def read_file(fname):
    mat = scipy.io.loadmat(fname)
    print(mat)
    
if __name__ == "__main__":
    read_file(file)
