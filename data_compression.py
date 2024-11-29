import argparse
import numpy as np
import torch
import os, os.path as osp
from pathlib import Path
from tqdm import tqdm
from .dataset import Compressor, CubeDecimalCompressor

def main(args):
    # create output directory, override if exists
    output_dir = Path(args.output_dir)

    compressors: list[Compressor] = [CubeDecimalCompressor()]
    output_data = []

    # iterate over all files in the input directory
    for file in tqdm(os.listdir(args.input_dir)):
        # load the data
        data = np.load(osp.join(args.input_dir, file))
        
        # compress the data
        for compressor in compressors:
            data = compressor.compress(data)

        # save the compressed data
        output_data.append(data)
    
    # concatenate all compressed data
    output_data = np.stack(output_data, axis=0)
    
    # save the tensor
    np.save(output_dir, output_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Compression')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    main(args)