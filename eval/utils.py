import os
import sys
import time
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


class PQ:
    def __init__(self, centers=None, n_clusters=256, dim=16):
        """Product quantization for better cluster
        args:
            n_clusters (int): number clusters on each product
            dim (int): dimension of each cluster
        """
        self.n_clusters = n_clusters
        self.dim = dim
        self.centers = centers
        self.kmeans = []
        for i in range(0, self.centers.shape[1], self.dim):
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.cluster_centers_ = self.centers[:, i:i+self.dim]
            self.kmeans.append(kmeans)

    def predict(self, V):
        """Predict an array vlad-ed
        params:
            V (array): a MxD array vlad-ed need to be predict
        returns:
            labels (array): a Mx(D/dim) array label
        """
        assert V.shape[1] == self.centers.shape[1]
        labels = np.zeros((V.shape[0], V.shape[1]//self.dim), dtype=np.uint8)
        for i in range(0, V.shape[1], self.dim):
            label = self.kmeans[i//self.dim].predict(V[:, i:i+self.dim])
            labels[:, i//self.dim] = label
        return labels

    def reconstruct(self, labels):
        """
        Reconstruct an array (predicted) by centers
        This method is only for testing (takes RAM to store values)
        params:
            labels (array): a Mx(D/dim) array contains label need to be reconstructed
        returns:
            V (array): a MxD array (vlad-ed like)
        """
        V = np.zeros((labels.shape[0], labels.shape[1]*self.dim), dtype=np.float32)
        for i in range(0, labels.shape[1]*self.dim, self.dim):
            V[:, i:i+self.dim] = self.centers[:, i: i+self.dim][labels[:, i//self.dim]]
        return V

    def quantize(self, V):
        labels = self.predict(V)
        V = self.reconstruct(labels)
        return V

    def save(self, h5py_path, raw_path):
        """Save centers into file
        params:
            h5py_path (str): where to save centers with h5py format
                            `dim`: dimension
                            `centers`: centers
            raw_path (str): save centers with raw format.
                            first line: dim
                            next lines: centers

        """
        with h5py.File(h5py_path, 'w') as hf:
            hf.create_dataset("dim", data=[self.dim], dtype="uint16")
            hf.create_dataset("centers", data=self.centers, dtype="float32")
        with open(raw_path, 'w') as f:
            for center in self.centers:
                strc = [str(c) for c in center]
                f.write(' '.join(strc) + '\n')


from sklearn.cluster import KMeans
class VLAD:
    def __init__(self, centers, norm=True):
        """VLAD
        args:
            centers (numpy array): Dx128 dimension centers
            norm (bool): use intra normalization
        """
        self.centers = centers
        self.kmeans = KMeans(n_clusters=int(self.centers.shape[0]))
        self.kmeans.cluster_centers_ = self.centers
        self.norm = norm

    def encode(self, desc=None):
        """Encode descriptors into vlad
        params:
            desc (numpy array): Nx8 descriptors of an images
        returns:
            Dx128 numpy array encoded by VLAD
        """
        if desc is None:
            return None
        labels = self.kmeans.predict(desc)
        v = np.zeros(self.centers.shape, dtype=np.float32)
        for i in range(v.shape[0]):
            if np.sum(labels==i) > 0:
                v[i] = np.sum(desc[labels==i,:] - self.centers[i],axis=0)
        if self.norm:
            norms = np.sqrt(np.sum(v**2, axis=1))
            norms[norms < 1e-12] = 1e-12
            v = v / norms.reshape(-1, 1)
        v = v.flatten()
        return v


class compactCode:
    def __init__(self, centers_path, pq_centers_path, codes_path, codes_name):

        with h5py.File(centers_path, 'r') as hf:
            centers = hf["centers"][:]
        with h5py.File(pq_centers_path, 'r') as hf:
            pq_centers = hf['centers'][:]

        with open(codes_name, 'r') as f:
            names = f.readlines()
            names = [name[:-1] for name in names]
            names = np.array(names)

        with open(codes_path, 'r') as f:
            codes = f.readlines()
            codes = [(code[:-1].strip()).split(' ') for code in codes]
            codes = np.array(codes, dtype=np.uint8)

        self.names = names
        self.codes = codes
        self.vlad = VLAD(centers, False)
        self.pq = PQ(pq_centers, 256, 64)


    def code(self, desc=None):
        if desc is None:
            return None
        return self.pq.predict(self.vlad.encode(desc).reshape(1, -1))

    def lookup_table(self, vlad_vector=None):
        assert vlad_vector is not None

        table = np.zeros((self.pq.centers.shape[0], self.pq.centers.shape[1]//self.pq.dim), dtype=np.float32)
        for i in range(table.shape[1]):
            cur_centers = self.pq.centers[:, i*self.pq.dim: (i+1)*self.pq.dim]
            cur_vector = vlad_vector[i*self.pq.dim: (i+1)*(self.pq.dim)]
            table[:, i] = np.sum((cur_centers - cur_vector)**2, axis=1)
        return table

    def search(self, desc):
        v = self.vlad.encode(desc)
        table = self.lookup_table(v)

        distance = np.zeros(self.codes.shape, dtype=np.float32)

        for i in range(self.codes.shape[1]):
            distance[:, i] = table[:, i][self.codes[:, i]]

        distance = distance.sum(axis=1)
        indices = distance.argsort()

        return self.names[indices], distance[indices]


class Search:
    def __init__(self, model):
        self.model = model

        #bbox
        self.x, self.y = None, None
        self.w, self.h = None, None

        self.image = None
        self.patch = None

        self.keys = None
        self.desc = None
        self.rank_list = None
        self.ratio = 0.85

        self.flann_params = dict(algorithm=1, trees=5)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})

    def search(self, image_dir, coords):

        self.x, self.y, self.w, self.h = coords
        self.image_dir = image_dir

        self.image = cv2.imread(self.image_dir)
        self.patch = self.image[self.y: self.y+self.h, self.x:self.x+self.w]

        cv2.imwrite('temp/temp.jpg', self.patch)
        os.system('../hesaff/extract_sift_from_image_quiet temp/temp.jpg temp/temp.sift')

        with open('temp/temp.sift', 'r') as f:
            text = f.readlines()
            values = [t[:-1].split(' ') for t in text[2:]]
            values = np.array(values, dtype=np.float32)
            self.keys = np.array(values[:, :5], dtype=np.float32)
            self.desc = np.array(values[:, 5:], dtype=np.uint8)

        self.rank_list, self.distance = self.model.search(self.desc)

