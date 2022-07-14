from dask.distributed import progress, Client, LocalCluster
import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd

def mmap_load_chunk(filename, shape, dtype, offset, sl):
    '''
    Memory map the given file with overall shape and dtype and return a slice
    specified by :code:`sl`.

    Parameters
    ----------

    filename : str
    shape : tuple
        Total shape of the data in the file
    dtype:
        NumPy dtype of the data in the file
    offset : int
        Skip :code:`offset` bytes from the beginning of the file.
    sl:
        Object that can be used for indexing or slicing a NumPy array to
        extract a chunk

    Returns
    -------

    numpy.memmap or numpy.ndarray
        View into memory map created by indexing with :code:`sl`,
        or NumPy ndarray in case no view can be created using :code:`sl`.
    '''
    data = np.memmap(filename, mode='r', shape=shape, dtype=dtype, offset=offset)
    return data[sl]


def mmap_dask_array(filename, shape, dtype, offset=0, blocksize=5):
    '''
    Create a Dask array from raw binary data in :code:`filename`
    by memory mapping.

    This method is particularly effective if the file is already
    in the file system cache and if arbitrary smaller subsets are
    to be extracted from the Dask array without optimizing its
    chunking scheme.

    It may perform poorly on Windows if the file is not in the file
    system cache. On Linux it performs well under most circumstances.

    Parameters
    ----------

    filename : str
    shape : tuple
        Total shape of the data in the file
    dtype:
        NumPy dtype of the data in the file
    offset : int, optional
        Skip :code:`offset` bytes from the beginning of the file.
    blocksize : int, optional
        Chunk size for the outermost axis. The other axes remain unchunked.

    Returns
    -------

    dask.array.Array
        Dask array matching :code:`shape` and :code:`dtype`, backed by
        memory-mapped chunks.
    '''
    load = dask.delayed(mmap_load_chunk)
    chunks = []
    for index in range(0, shape[0], blocksize):
        # Truncate the last chunk if necessary
        chunk_size = min(blocksize, shape[0] - index)
        chunk = dask.array.from_delayed(
            load(
                filename,
                shape=shape,
                dtype=dtype,
                offset=offset,
                sl=slice(index, index + chunk_size)
            ),
            shape=(chunk_size, ) + shape[1:],
            dtype=dtype
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=0)

if __name__ == "__main__":

    with dask.config.set({'temporary_directory': '/home/yorai/Wiki/'}):
        cluster = LocalCluster()
        client = Client(cluster)
        path = "/mnt/nvme/btrfs/enwiki-20220601-pages-articles-multistream.xml-raw"
        txt = np.memmap(path,mode='r',dtype=np.int64)
        hist89G=dd.from_array(txt).min()
        
        hist89GArray = mmap_dask_array(
            filename=path,
            shape=(len(txt),1),
            dtype=np.int64,
            blocksize = 16*1024*1024
            # real    5m49.969s
            # user    51m3.762s
            # sys     4m10.824s
        )
        hist89G = dd.from_dask_array(hist89GArray).map_partitions(lambda x: pd.DataFrame.from_dict(dict(zip(*(np.unique(x, return_counts=True)))), orient='index').reset_index().set_axis(['key', 'cnt'], axis='columns')).groupby(by='key').sum()
        hist89G.to_csv('89G-*.csv')