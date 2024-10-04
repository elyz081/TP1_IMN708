#! /usr/bin/env python

"""
Description of what the script does
"""

import argparse
import nibabel as nib
import tools.math
import tools.utils
import tools.io
import tools.display
import matplotlib.pyplot as plt
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('axe', type=int, default=0,
                        help='Axe de la vue (Sagittale 0, Coronale 1, Axiale 2)')
    

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_image)
    img = image.get_fdata()
    data = tools.io.reorient_data_rsa(image)
    voxel_sizes = image.header.get_zooms()
    tools.display.display_image(data, voxel_sizes, args.axe)
    MIP= tools.math.MIP(img, args.axe)
    #TODO display the mip
    fig, ax = plt.subplots()
    ax.imshow(MIP, cmap='gray')
    ax.set_title("MIP")
    plt.show()  


if __name__ == "__main__":
    main()