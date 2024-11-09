#!/usr/bin/env python
"""
This script calculates and displays the joint histogram of two grayscale images
along with similarity metrics (SSD, Correlation Ratio, and Mutual Information).

Features:
---------
- Computes and displays the joint histogram for two input images using a specified number of bins.
- Calculates similarity metrics such as SSD, Correlation Ratio, and Mutual Information.
- Supports visualization using either Plotly (default) or Matplotlib (optional).

Usage:
------
Example of running the script:

    python show_joint_hist.py <in_image_1> <in_image_2> --bins <num_bins> [--plt]

Parameters:
-----------
in_image_1: str
    Path to the first input image.
in_image_2: str
    Path to the second input image.
--bins: int, optional
    Number of bins to use for the joint histogram. Default is 256.
--plt: flag, optional
    If specified, uses Matplotlib for displaying the histograms (default is False).
"""

import argparse
from tools import io, display, math
import matplotlib.pyplot as plt
import numpy as np


def _build_arg_parser():
    """
    Builds and returns the argument parser for the script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=__doc__)

    # Required positional arguments for input images
    parser.add_argument('in_image_1', help='Path to the first input image.')
    parser.add_argument('in_image_2', help='Path to the second input image.')

    # Optional arguments
    parser.add_argument('--bins', type=int, default=256,
                        help='Number of bins for the joint histogram.')
    parser.add_argument('--plt', action='store_true',
                        help='Display using Matplotlib instead of Plotly.')
    return parser

def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the input images
    data1 = io.image_to_data(args.in_image_1)
    data2 = io.image_to_data(args.in_image_2)

    # Convert the images to bins
    data_bins_1, bin_centers_1 = io.data_to_bins(data1, args.bins)
    data_bins_2, bin_centers_2 = io.data_to_bins(data2, args.bins)

    # Compute the joint histogram
    joint_hist = math.joint_histogram(data_bins_1, data_bins_2)

    # Calculate similarity metrics
    ssd = math.ssd(data1, data2)
    ssd_jh = math.normalized_ssd(joint_hist, bin_centers_1, bin_centers_2)
    cr = math.cr(joint_hist)
    im = math.IM(joint_hist)

    # Display the joint histogram and metrics using the specified visualization method
    if args.plt:
        # Use Matplotlib
        display.plt_display_joint_hist(joint_hist, data1, data2, args.bins, ssd, ssd_jh, cr, im)
    else:
        # Use Plotly (default)
        display.display_joint_hist(joint_hist, bin_centers_1, bin_centers_2, data1, data2, args.bins, ssd, ssd_jh, cr, im)


if __name__ == "__main__":
    main()
