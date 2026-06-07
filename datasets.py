import gzip
import shutil
import math
import numpy
import os
import random
import subprocess
import sys
import struct
import time

import numpy as np
from scipy.sparse import csr_matrix

from urllib.request import urlretrieve

from dataset_io import (
    xbin_mmap, download_accelerated, download, sanitize,
    knn_result_read, range_result_read, read_sparse_matrix,
    write_sparse_matrix, fvecs_read, bvecs_mmap, ivecs_read
)


BASEDIR = "data/"


def write_fvecs(path, vectors):
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("write_fvecs expects a 2-D array")
    n, dim = vectors.shape
    dtype = np.dtype([("dim", np.int32), ("vec", np.float32, (dim,))])
    packed = np.empty(n, dtype=dtype)
    packed["dim"] = dim
    packed["vec"] = vectors
    packed.tofile(path)


def write_bvecs(path, vectors):
    vectors = np.ascontiguousarray(vectors, dtype=np.uint8)
    if vectors.ndim != 2:
        raise ValueError("write_bvecs expects a 2-D array")
    n, dim = vectors.shape
    dtype = np.dtype([("dim", np.int32), ("vec", np.uint8, (dim,))])
    packed = np.empty(n, dtype=dtype)
    packed["dim"] = dim
    packed["vec"] = vectors
    packed.tofile(path)


def write_ivecs(path, vectors):
    vectors = np.ascontiguousarray(vectors, dtype=np.int32)
    if vectors.ndim != 2:
        raise ValueError("write_ivecs expects a 2-D array")
    n, dim = vectors.shape
    dtype = np.dtype([("dim", np.int32), ("vec", np.int32, (dim,))])
    packed = np.empty(n, dtype=dtype)
    packed["dim"] = dim
    packed["vec"] = vectors
    packed.tofile(path)


class Dataset():
    def prepare(self):
        """
        Download and prepare dataset, queries, groundtruth.
        """
        pass
    def get_dataset_fn(self):
        """
        Return filename of dataset file.
        """
        pass
    def get_dataset(self):
        """
        Return memmapped version of the dataset.
        """
        pass
    def get_dataset_iterator(self, bs=512, split=(1, 0)):
        """
        Return iterator over blocks of dataset of size at most 512.
        The split argument takes a pair of integers (n, p) where p = 0..n-1
        The dataset is split in n shards, and the iterator returns only shard #p
        This makes it possible to process the dataset independently from several
        processes / threads.
        """
        pass
    def get_queries(self):
        """
        Return (nq, d) array containing the nq queries.
        """
        pass
    def get_private_queries(self):
        """
        Return (private_nq, d) array containing the private_nq private queries.
        """
        pass
    def get_groundtruth(self, k=None):
        """
        Return (nq, k) array containing groundtruth indices
        for each query."""
        pass

    def search_type(self):
        """
        "knn" or "range" or "knn_filtered"
        """
        pass

    def distance(self):
        """
        "euclidean" or "ip" or "angular"
        """
        pass

    def data_type(self):
        """
        "dense" or "sparse"
        """
        pass

    def default_count(self):
        """ number of neighbors to return """
        return 10

    def short_name(self):
        return f"{self.__class__.__name__}-{self.nb}"
    
    def __str__(self):
        return (
            f"Dataset {self.__class__.__name__} in dimension {self.d}, with distance {self.distance()}, "
            f"search_type {self.search_type()}, size: Q {self.nq} B {self.nb}")

    def get_random_subset(self, size, seed):
        np.random.seed(seed)
        ds = self.get_dataset()
        indices = np.random.choice(ds.shape[0], size, replace=False)
        return ds[indices]


class DatasetCompetitionFormat(Dataset):
    """
    Dataset in the native competition format, that is able to read the
    files in the https://big-ann-benchmarks.com/ page.
    The constructor should set all fields. The functions below are generic.

    For the 10M versions of the dataset, the database files are downloaded in
    part and stored with a specific suffix. This is to avoid having to maintain
    two versions of the file.
    """

    def prepare(self, skip_data=False, original_size=10**9):
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        # start with the small ones...
        for fn in [self.qs_fn, self.gt_fn]:
            if fn is None:
                continue
            if fn.startswith("https://"):
                sourceurl = fn
                outfile = os.path.join(self.basedir, fn.split("/")[-1])
            else:
                sourceurl = os.path.join(self.base_url, fn)
                outfile = os.path.join(self.basedir, fn)
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
                continue
            download(sourceurl, outfile)

        # private qs url
        if self.private_qs_url:
            outfile = os.path.join(self.basedir, self.private_qs_url.split("/")[-1])
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
            else:
                download(self.private_qs_url, outfile)

        # private gt url
        if self.private_gt_url:
            outfile = os.path.join(self.basedir, self.private_gt_url.split("/")[-1])
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
            else:
                download(self.private_gt_url, outfile)

        if skip_data:
            return

        fn = self.ds_fn
        sourceurl = os.path.join(self.base_url, fn)
        outfile = os.path.join(self.basedir, fn)
        if self.nb != original_size:
            outfile = outfile + '.crop_nb_%d' % self.nb
            
        if os.path.exists(outfile):
            print("file %s already exists" % outfile)
            return
        if self.nb == 10**9:
            download_accelerated(sourceurl, outfile)
        elif self.nb == original_size:
            #if nb vectors is less than 1 billion, can download the whole dataset in the normal fashion without a cropped header
            download(sourceurl, outfile)
        else:
            # download cropped version of file
            file_size = 8 + self.d * self.nb * np.dtype(self.dtype).itemsize
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
                return
            download(sourceurl, outfile, max_size=file_size)
            # then overwrite the header...
            header = np.memmap(outfile, shape=2, dtype='uint32', mode="r+")
            assert header[0] == original_size
            assert header[1] == self.d
            header[0] = self.nb

    def get_dataset_fn(self):
        fn = os.path.join(self.basedir, self.ds_fn)
        if os.path.exists(fn):
            return fn
        else:
            raise RuntimeError("file %s not found" %fn)

    def get_dataset_iterator(self, bs=512, split=(1,0)):
        nsplit, rank = split
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        filename = self.get_dataset_fn()
        x = xbin_mmap(filename, dtype=self.dtype, maxn=self.nb)
        assert x.shape == (self.nb, self.d)
        for j0 in range(i0, i1, bs):
            j1 = min(j0 + bs, i1)
            yield sanitize(x[j0:j1])

    def get_data_in_range(self, start, end):
        assert start >= 0
        assert end <= self.nb
        filename = self.get_dataset_fn()
        x = xbin_mmap(filename, dtype=self.dtype, maxn=self.nb)
        return x[start:end]

    def search_type(self):
        return "knn"
    
    def data_type(self):
        return "dense"

    def get_groundtruth(self, k=None):
        assert self.gt_fn is not None
        fn = self.gt_fn.split("/")[-1]   # in case it's a URL
        assert self.search_type() in ("knn", "knn_filtered")

        I, D = knn_result_read(os.path.join(self.basedir, fn))
        assert I.shape[0] == self.nq
        if k is not None:
            assert k <= 100
            I = I[:, :k]
            D = D[:, :k]
        return I, D

    def get_dataset(self):
        assert self.nb <= 10**7, "dataset too large, use iterator"
        slice = next(self.get_dataset_iterator(bs=self.nb))
        return sanitize(slice)

    def get_queries(self):
        filename = os.path.join(self.basedir, self.qs_fn)
        x = xbin_mmap(filename, dtype=self.dtype)
        assert x.shape == (self.nq, self.d)
        return sanitize(x)

    def get_private_queries(self):
        filename = os.path.join(self.basedir, self.qs_private_fn)
        x = xbin_mmap(filename, dtype=self.dtype)
        assert x.shape == (self.private_nq, self.d)
        return sanitize(x)

    def get_private_groundtruth(self, k=None):
        assert self.private_gt_fn is not None
        assert self.search_type() in ("knn", "knn_filtered")

        I, D = knn_result_read(os.path.join(self.basedir, self.private_gt_fn))
        assert I.shape[0] == self.private_nq
        if k is not None:
            assert k <= 100
            I = I[:, :k]
            D = D[:, :k]
        return I, D


subset_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/"


class RandomDS(DatasetCompetitionFormat):
    def __init__(self, nb, nq, d, basedir="random"):
        self.nb = nb
        self.nq = nq
        self.d = d
        self.dtype = 'float32'
        self.ds_fn = f"data_{self.nb}_{self.d}"
        self.qs_fn = f"queries_{self.nq}_{self.d}"
        self.gt_fn = f"gt_{self.nb}_{self.nq}_{self.d}"
        self.basedir = os.path.join(BASEDIR, f"{basedir}{self.nb}")
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

    def prepare(self, skip_data=False):
        import sklearn.datasets
        import sklearn.model_selection
        from sklearn.neighbors import NearestNeighbors

        print(f"Preparing datasets with {self.nb} random points and {self.nq} queries.")


        X, _ = sklearn.datasets.make_blobs(
            n_samples=self.nb + self.nq, n_features=self.d,
            centers=self.nq, random_state=1)

        data, queries = sklearn.model_selection.train_test_split(
            X, test_size=self.nq, random_state=1)


        with open(os.path.join(self.basedir, self.ds_fn), "wb") as f:
            np.array([self.nb, self.d], dtype='uint32').tofile(f)
            data.astype('float32').tofile(f)
        with open(os.path.join(self.basedir, self.qs_fn), "wb") as f:
            np.array([self.nq, self.d], dtype='uint32').tofile(f)
            queries.astype('float32').tofile(f)

        print("Computing groundtruth")

        nbrs = NearestNeighbors(n_neighbors=100, metric="euclidean", algorithm='brute').fit(data)
        D, I = nbrs.kneighbors(queries)
        with open(os.path.join(self.basedir, self.gt_fn), "wb") as f:
            np.array([self.nq, 100], dtype='uint32').tofile(f)
            I.astype('uint32').tofile(f)
            D.astype('float32').tofile(f)

    def search_type(self):
        return "knn"

    def distance(self):
        return "euclidean"

    def __str__(self):
        return f"Random({self.nb})"

    def default_count(self):
        return 10
    

class YFCC100MDataset(DatasetCompetitionFormat):
    """ the 2023 competition """

    def __init__(self, filtered=True, dummy=False):
        self.filtered = filtered
        nb_M = 10
        self.nb_M = nb_M
        self.nb = 10**6 * nb_M
        self.d = 192
        self.nq = 100000
        self.dtype = "uint8"
        private_key = 2727415019
        self.gt_private_fn = ""

        if dummy:
            # for now it's dummy because we don't have the descriptors yet
            self.ds_fn = "dummy2.base.10M.u8bin"
            self.qs_fn = "dummy2.query.public.100K.u8bin"
            self.qs_private_fn = "dummy2.query.private.%d.100K.u8bin" % private_key
            self.ds_metadata_fn = "dummy2.base.metadata.10M.spmat"
            self.qs_metadata_fn = "dummy2.query.metadata.public.100K.spmat"
            self.qs_private_metadata_fn = "dummy2.query.metadata.private.%d.100K.spmat" % private_key
            if filtered:
                # no subset as the database is pretty small.
                self.gt_fn = "dummy2.GT.public.ibin"
            else:
                self.gt_fn = "dummy2.unfiltered.GT.public.ibin"

        else:
            # with Zilliz' CLIP descriptors
            self.ds_fn = "base.10M.u8bin"
            self.qs_fn = "query.public.100K.u8bin"
            self.qs_private_fn = "query.private.%d.100K.u8bin" % private_key
            self.ds_metadata_fn = "base.metadata.10M.spmat"
            self.qs_metadata_fn = "query.metadata.public.100K.spmat"
            self.qs_private_metadata_fn = "query.metadata.private.%d.100K.spmat" % private_key
            if filtered:
                # no subset as the database is pretty small.
                self.gt_fn = "GT.public.ibin"
                self.gt_private_fn = "GT.private.%d.ibin" % private_key
            else:
                self.gt_fn = "unfiltered.GT.public.ibin"      

            self.private_gt_fn = "GT.private.%d.ibin" % private_key

            # data is uploaded but download script not ready.
        self.base_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
        self.basedir = os.path.join(BASEDIR, "yfcc100M")

        self.private_nq = 100000
        self.private_qs_url = self.base_url + self.qs_private_fn
        self.private_gt_url = self.base_url + self.gt_private_fn

        self.metadata_base_url = self.base_url + self.ds_metadata_fn
        self.metadata_queries_url = self.base_url + self.qs_metadata_fn
        self.metadata_private_queries_url = self.base_url + self.qs_private_metadata_fn

    def prepare(self, skip_data=False):
        super().prepare(skip_data, 10**7)
        for fn in (self.metadata_base_url, self.metadata_queries_url, 
                   self.metadata_private_queries_url):
            if fn:
                outfile = os.path.join(self.basedir, fn.split("/")[-1])
                if os.path.exists(outfile):
                    print("file %s already exists" % outfile)
                else:
                    download(fn, outfile)

    def set_subset(self, size):
        self.original_nb = self.nb
        self.nb = size
        self.original_basedir = self.basedir
        self.basedir = os.path.join(BASEDIR, f"yfcc100M_sampled_{size}")
        self.ds_fn = f"base.{size}.u8bin"
        self.ds_metadata_fn = f"base.metadata.{size}.spmat"

    def prepare_subset(self):
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        # Ensure original data is ready
        full_ds = YFCC100MDataset(filtered=self.filtered)
        full_ds.prepare(skip_data=False)
        
        # Create subset dataset
        src_ds = os.path.join(full_ds.basedir, full_ds.ds_fn)
        dst_ds = os.path.join(self.basedir, self.ds_fn)
        
        if not os.path.exists(dst_ds):
            print(f"Creating subset dataset: {dst_ds}")
            with open(src_ds, "rb") as f:
                n, d = np.fromfile(f, dtype='uint32', count=2)
                data = np.fromfile(f, dtype=self.dtype, count=self.nb * d)
                data = data.reshape(self.nb, d)
            
            with open(dst_ds, "wb") as f:
                np.array([self.nb, d], dtype='uint32').tofile(f)
                data.tofile(f)
                
        # Create subset metadata
        src_meta = os.path.join(full_ds.basedir, full_ds.ds_metadata_fn)
        dst_meta = os.path.join(self.basedir, self.ds_metadata_fn)
        if not os.path.exists(dst_meta):
             print(f"Creating subset metadata: {dst_meta}")
             mat = read_sparse_matrix(src_meta)
             mat_subset = mat[:self.nb]
             write_sparse_matrix(mat_subset, dst_meta)
             
        # Copy other files (Queries, GT)
        # Note: We do NOT copy the ground truth (gt_fn) because the original GT 
        # corresponds to the full dataset indices. For the subset, we want the 
        # C++ backend to recompute the ground truth via brute force.
        for fn in [full_ds.qs_fn, full_ds.qs_metadata_fn]:
            if not fn: continue
            src = os.path.join(full_ds.basedir, fn)
            dst = os.path.join(self.basedir, fn)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)

    def get_dataset_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.ds_metadata_fn))

    def get_queries_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.qs_metadata_fn))
    
    def get_private_queries_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.qs_private_metadata_fn))
    
    def distance(self):
        return "euclidean"

    def search_type(self):
        if self.filtered:
            return "knn_filtered"
        else:
            return "knn"


class Sift1MDataset(Dataset):
    def __init__(self):
        self.d = 128
        self.nb = 1000000
        self.nq = 10000
        self.basedir = os.path.join(BASEDIR, "sift1m")
        self.base_fn = "sift_base.fvecs"
        self.query_fn = "sift_query.fvecs"
        self.gt_fn = "sift_groundtruth.ivecs"
        self.url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

    def prepare(self):
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)
        
        base_path = os.path.join(self.basedir, self.base_fn)
        query_path = os.path.join(self.basedir, self.query_fn)
        gt_path = os.path.join(self.basedir, self.gt_fn)

        if os.path.exists(base_path) and os.path.exists(query_path) and os.path.exists(gt_path):
            return

        import tarfile
        
        tar_path = os.path.join(self.basedir, "sift.tar.gz")
        if not os.path.exists(tar_path):
            print("Downloading SIFT1M...")
            download(self.url, tar_path)
        
        print("Extracting SIFT1M...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.basedir)
        
        # Move files if they are in a subdirectory (sift/...)
        sift_dir = os.path.join(self.basedir, "sift")
        if os.path.exists(sift_dir):
            for f in ["sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"]:
                src = os.path.join(sift_dir, f)
                dst = os.path.join(self.basedir, f)
                if os.path.exists(src):
                    shutil.move(src, dst)
            try:
                os.rmdir(sift_dir)
            except OSError:
                print(f"Warning: Could not remove directory {sift_dir}, it might not be empty.")

    def get_dataset(self):
        return fvecs_read(os.path.join(self.basedir, self.base_fn))

    def get_queries(self):
        return fvecs_read(os.path.join(self.basedir, self.query_fn))

    def get_groundtruth(self, k=None):
        from .dataset_io import ivecs_read
        I = ivecs_read(os.path.join(self.basedir, self.gt_fn))
        if k is not None:
            I = I[:, :k]
        return I, None

    def search_type(self):
        return "knn"

    def distance(self):
        return "euclidean"

    def data_type(self):
        return "dense"


class Sift100kSampleDataset(Dataset):
    def __init__(self, target_nb=100000, topk=10, seed=123):
        self.d = 128
        self.nq = 10000
        self.nb = target_nb
        self.topk = topk
        self.seed = seed
        self.target_nb = target_nb
        self.basedir = os.path.join(BASEDIR, f"sift1m_sample_{target_nb}")
        self.base_fn = "base.fvecs"
        self.query_fn = "query.fvecs"
        self.gt_fn = "groundtruth.ivecs"
        self.base_attrs_path = os.path.join(self.basedir, "base_attrs.bvecs")
        self.query_attrs_path = os.path.join(self.basedir, "query_attrs.bvecs")

    def prepare(self):
        os.makedirs(self.basedir, exist_ok=True)
        required = [
            os.path.join(self.basedir, self.base_fn),
            os.path.join(self.basedir, self.query_fn),
            os.path.join(self.basedir, self.gt_fn),
            self.base_attrs_path,
            self.query_attrs_path,
        ]
        if all(os.path.exists(path) for path in required):
            return

        full = Sift1MDataset()
        full.prepare()

        source_base = os.path.join(full.basedir, full.base_fn)
        source_query = os.path.join(full.basedir, full.query_fn)
        source_gt = os.path.join(full.basedir, full.gt_fn)

        base_attrs_src, query_attrs_src = self._ensure_attribute_bitmaps(full.basedir)

        gt = ivecs_read(source_gt)
        if gt.shape[0] != self.nq:
            raise ValueError("Unexpected query count in SIFT ground truth")
        gt = gt[:, :self.topk]
        primary = np.unique(gt.reshape(-1))
        target = max(self.target_nb, primary.size)

        if primary.size < target:
            all_indices = np.arange(full.nb, dtype=np.int64)
            mask = np.ones(full.nb, dtype=bool)
            mask[primary] = False
            pool = all_indices[mask]
            if pool.size < (target - primary.size):
                raise ValueError("Insufficient leftover base vectors to pad sample")
            rng = np.random.default_rng(self.seed)
            extra = rng.choice(pool, size=target - primary.size, replace=False)
            selected = np.concatenate([primary, extra])
        else:
            selected = primary
        selected = np.sort(selected)
        self.nb = selected.size

        mapping = np.full(full.nb, -1, dtype=np.int64)
        mapping[selected] = np.arange(selected.size)
        remapped_gt = mapping[gt]
        if np.any(remapped_gt < 0):
            raise ValueError("Ground truth references missing from sampled base")
        remapped_gt = remapped_gt.astype(np.int32, copy=False)

        base_vectors = fvecs_read(source_base)
        base_subset = np.ascontiguousarray(base_vectors[selected])
        del base_vectors

        base_attrs_full = np.array(bvecs_mmap(base_attrs_src), copy=True)
        base_attrs_subset = base_attrs_full[selected]

        write_fvecs(os.path.join(self.basedir, self.base_fn), base_subset)
        write_bvecs(self.base_attrs_path, base_attrs_subset)
        write_ivecs(os.path.join(self.basedir, self.gt_fn), remapped_gt)

        self._copy_file(source_query, os.path.join(self.basedir, self.query_fn))
        self._copy_file(query_attrs_src, self.query_attrs_path)

    def _ensure_attribute_bitmaps(self, sift_dir):
        base_attrs = os.path.join(sift_dir, "sift_base_attrs.bvecs")
        query_attrs = os.path.join(sift_dir, "sift_query_attrs.bvecs")
        if os.path.exists(base_attrs) and os.path.exists(query_attrs):
            return base_attrs, query_attrs

        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        generator = os.path.join(script_dir, "process_sift1m.py")
        raw_dir = os.path.join(sift_dir, "raw")
        if not os.path.exists(raw_dir):
            raw_dir = sift_dir
        subprocess.run([
            sys.executable,
            generator,
            "--raw-dir", raw_dir,
            "--out-dir", sift_dir,
            "--force",
        ], check=True)
        if not (os.path.exists(base_attrs) and os.path.exists(query_attrs)):
            raise RuntimeError("Failed to generate SIFT attribute bitmaps")
        return base_attrs, query_attrs

    def _copy_file(self, src, dst):
        shutil.copy2(src, dst)

    def search_type(self):
        return "knn"

    def distance(self):
        return "euclidean"

    def data_type(self):
        return "dense"


def _strip_gz(filename):
    if not filename.endswith('.gz'):
        raise RuntimeError(f"expected a filename ending with '.gz'. Received: {filename}")
    return filename[:-3]


def _gunzip_if_needed(filename):
    if filename.endswith('.gz'):
        print('unzipping', filename, '...', end=" ", flush=True)

        with gzip.open(filename, 'rb') as f_in, open(_strip_gz(filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(filename)
        print('done.')



class RandomFilterDS(RandomDS):
    def __init__(self, nb, nq, d):
        super().__init__(nb, nq, d, "random-filter")
        self.ds_metadata_fn = f"data_metadata_{self.nb}_{self.d}"
        self.qs_metadata_fn = f"queries_metadata_{self.nb}_{self.d}"

    def prepare(self, skip_data=False):
        import sklearn.datasets
        import sklearn.model_selection
        from sklearn.neighbors import NearestNeighbors

        print(f"Preparing datasets with {self.nb} random points, {self.nq} queries, and two filters.")

        X, _ = sklearn.datasets.make_blobs(
            n_samples=self.nb + self.nq, n_features=self.d,
            centers=self.nq, random_state=1)

        data, queries = sklearn.model_selection.train_test_split(
            X, test_size=self.nq, random_state=1) 

        filter1 = [1, 2]
        filter2 = [3, 4]       

        assert self.nb % 2 == 0

        # simple filters, first half of the data matches second 
        # half of the queries, and vice versa

        data_filters = [filter1] * (self.nb // 2) + [filter2] * (self.nb // 2)
        query_filters = [filter2] * (self.nq // 2) + [filter1] * (self.nq // 2)

        assert len(data_filters) == data.shape[0]

        with open(os.path.join(self.basedir, self.ds_fn), "wb") as f:
            np.array([self.nb, self.d], dtype='uint32').tofile(f)
            data.astype('float32').tofile(f)
        with open(os.path.join(self.basedir, self.qs_fn), "wb") as f:
            np.array([self.nq, self.d], dtype='uint32').tofile(f)
            queries.astype('float32').tofile(f) 

        data_indices = np.array(data_filters).flatten()
        data_indptr = [2 * i for i in range(self.nb)] + [2 * self.nb]
        data_data = [1] * self.nb * 2
        data_metadata_sparse = csr_matrix((data_data, data_indices, data_indptr))

        query_indices = np.array(query_filters).flatten()
        query_indptr = [2 * i for i in range(self.nq)] + [2 * self.nq]
        query_data = [1] * self.nq * 2
        query_metadata_sparse = csr_matrix((query_data, query_indices, query_indptr))

        write_sparse_matrix(data_metadata_sparse, 
                            os.path.join(self.basedir, self.ds_metadata_fn))
        write_sparse_matrix(query_metadata_sparse, 
                            os.path.join(self.basedir, self.qs_metadata_fn))

        print("Computing groundtruth")

        n_neighbors = 100

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm='brute').fit(data[:self.nb // 2])
        DD, II = nbrs.kneighbors(queries[self.nq // 2:])

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm='brute').fit(data[self.nb // 2: ])
        D, I = nbrs.kneighbors(queries[:self.nq // 2])

        D = np.concatenate((D, DD))
        I = np.concatenate((I + self.nb // 2, II))

        with open(os.path.join(self.basedir, self.gt_fn), "wb") as f:
            np.array([self.nq, n_neighbors], dtype='uint32').tofile(f)
            I.astype('uint32').tofile(f)
            D.astype('float32').tofile(f)

    def get_dataset_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.ds_metadata_fn))

    def get_queries_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.qs_metadata_fn))
    
    def search_type(self):
        return "knn_filtered"

    def __str__(self):
        return f"RandomFilter({self.nb})"



DATASETS = {
    'yfcc-10M': lambda: YFCC100MDataset(),
    'yfcc-10M-unfiltered': lambda: YFCC100MDataset(filtered=False),
    'yfcc-10M-dummy': lambda: YFCC100MDataset(dummy=True),
    'yfcc-10M-dummy-unfiltered': lambda: YFCC100MDataset(filtered=False, dummy=True),
    # 'yfcc-1M': lambda: YFCC1MDataset(),

    # 'sparse-small': lambda: SparseDataset("small"),
    # 'sparse-1M': lambda: SparseDataset("1M"),
    # 'sparse-full': lambda: SparseDataset("full"), 

    # 'wikipedia-35M': lambda : WikipediaDataset(35000000),
    # 'wikipedia-10M': lambda : WikipediaDataset(10000000),
    # 'wikipedia-1M': lambda : WikipediaDataset(1000000),
    # 'wikipedia-100K': lambda : WikipediaDataset(100000),

    # 'msmarco-100M': lambda : MSMarcoWebSearchDataset(101070374),
    # 'msmarco-10M': lambda : MSMarcoWebSearchDataset(10000000),
    # 'msmarco-1M': lambda : MSMarcoWebSearchDataset(1000000),

    # 'openai-2M': lambda : OpenAIArXivDataset(2321096),
    # 'openai-100K': lambda : OpenAIArXivDataset(100000),

    # 'random-xs': lambda : RandomDS(10000, 1000, 20),
    # 'random-s': lambda : RandomDS(100000, 1000, 50),

    # 'random-xs-clustered': lambda: RandomClusteredDS(),

    # 'random-range-xs': lambda : RandomRangeDS(10000, 1000, 20),
    # 'random-range-s': lambda : RandomRangeDS(100000, 1000, 50),

    # 'random-filter-s': lambda : RandomFilterDS(100000, 1000, 50),

    # 'openai-embedding-1M': lambda: OpenAIEmbedding1M(93652),
    'sift-1M': lambda: Sift1MDataset(),
    'sift-100k-sample': lambda: Sift100kSampleDataset(),
}
