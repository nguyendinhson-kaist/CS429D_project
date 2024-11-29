import numpy as np
import os
from multiprocessing import Pool

categories = ['airplane', 'chair', 'table']
split = ['train', 'test', 'val']
root = './data/shapenet'  # Define your root output directory

def process_split_category(args):
    sp, cat = args
    print(f'Processing {cat} {sp}...')
    
    # Load the data of shape [N, H, W, D]
    data = np.load(f'./data/hdf5_data/{cat}_voxels_{sp}.npy')
    
    # Define the output directory for this category and split
    output_dir = os.path.join(root, sp, cat)
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each N dimension and save it as an individual .npy file
    for i in range(data.shape[0]):
        # Extract the i-th element of shape [H, W, D]
        individual_data = data[i]
        
        # Define a file path for the individual file
        output_file = os.path.join(output_dir, f'{cat}_{sp}_{i}.npy')
        
        # Save the individual array
        np.save(output_file, individual_data)
    
    print(f'Finished processing {cat} {sp}.')

if __name__ == '__main__':
    # Create a list of all combinations of split and category to process
    tasks = [(sp, cat) for sp in split for cat in categories]
    
    # Create a pool of workers
    with Pool() as pool:
        # Map the tasks to the pool of workers
        pool.map(process_split_category, tasks)
