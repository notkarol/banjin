#!/usr/bin/env python3

import argparse
import banjin
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tiles", default=64, type=int,
                        help="Number of tiles per image to display")
    parser.add_argument("--n_pages", default=64, type=int,
                        help="Max number of images to display. 0 for all")
    parser.add_argument("--save", default=False, action='store_true',
                        help="Store tiles instead of display")
    parser.add_argument("filename", help="Name of pickle file")
    return parser.parse_args()

def main():

    # Initialize arguments and parameters
    args = arguments()
    with open(args.filename, 'rb') as f:
        X, Y = pickle.load(f)
    counter = 0
    
    # Prepare for saving if we need to
    if args.save:
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        dataname = os.path.splitext(os.path.basename(args.filename))[0]
        
    # Iterate through all cutout pages
    for out in banjin.getCutouts(X, args.n_tiles, Y):

        counter += 1
        if args.n_pages > 0 and counter > args.n_pages:
            break
        
        # Plot current page
        plt.imshow(out, interpolation="nearest", vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
        plt.title("Page %i" % counter)
        plt.axis('off')
        
        # Either save a file or display to screen
        if args.save:
            name = "plots/%s-%03i.png" % (dataname, counter)
            plt.savefig(name, bbox_inches='tight', dpi=160)
            print("Saving", name)
        else:
            plt.show()

if __name__ == "__main__":
    main()
