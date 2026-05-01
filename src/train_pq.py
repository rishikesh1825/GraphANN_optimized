import numpy as np
from sklearn.cluster import MiniBatchKMeans
import struct
import time

def read_fvecs(filename):
    print(f"Loading {filename}...")
    # .fvecs format: [dim (int32)] [float32 * dim]
    data = np.fromfile(filename, dtype=np.int32)
    dim = data[0]
    # Reshape and slice off the dimension header for each vector
    vectors = data.reshape(-1, dim + 1)[:, 1:].view(np.float32)
    return vectors, dim

def train_and_compress_pq():
    base_file = '../../sift/sift_base.fvecs'
    vectors, dim = read_fvecs(base_file)
    npts = vectors.shape[0]
    
    M = 32          # Number of sub-vectors
    K = 256         # Centroids per sub-vector (fits in uint8)
    chunk_dim = dim // M
    
    print(f"Dataset: {npts} points, {dim} dimensions.")
    print(f"PQ Parameters: M={M}, K={K}, Chunk Dimension={chunk_dim}")
    
    # Pre-allocate arrays
    codebook = np.zeros((M, K, chunk_dim), dtype=np.float32)
    pq_compressed = np.zeros((npts, M), dtype=np.uint8)
    
    # We train on a subset of 100,000 points to save time (standard practice)
    train_size = min(npts, 100000)
    train_data = vectors[:train_size]
    
    print("\nTraining Codebooks and Compressing Data...")
    start_time = time.time()
    
    for m in range(M):
        print(f"  Processing chunk {m+1}/{M}...")
        # Extract the specific sub-vector chunk across all data
        chunk_train = train_data[:, m * chunk_dim : (m + 1) * chunk_dim]
        chunk_all = vectors[:, m * chunk_dim : (m + 1) * chunk_dim]
        
        # Train K-Means
        kmeans = MiniBatchKMeans(n_clusters=K, batch_size=10000, max_iter=100, n_init=3, random_state=42)
        kmeans.fit(chunk_train)
        
        # Save the centroids to the codebook
        codebook[m] = kmeans.cluster_centers_
        
        # Encode the ENTIRE dataset using the trained centroids
        pq_compressed[:, m] = kmeans.predict(chunk_all)
        
    print(f"PQ Training & Compression finished in {time.time() - start_time:.2f} seconds.")
    
    # Save the Codebook (32 KB)
    with open('pq_codebook.bin', 'wb') as f:
        f.write(struct.pack('II', M, dim))
        f.write(codebook.tobytes())
        
    # Save the Compressed Data (16 MB)
    with open('pq_compressed.bin', 'wb') as f:
        f.write(struct.pack('II', npts, M))
        f.write(pq_compressed.tobytes())
        
    print("Saved pq_codebook.bin and pq_compressed.bin successfully!")

if __name__ == '__main__':
    train_and_compress_pq()