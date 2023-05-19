#!/usr/bin/env python3
# coding=utf-8
import argparse

import my_auto_alignment
import my_projector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Image to be aligned. Multiple files separated by ,')

    args = parser.parse_args()
    print(args)

    all_aligned_filenames = my_auto_alignment.main('align', args.source)
    all_projected_filenames = my_projector.main(','.join(all_aligned_filenames), auto_aligned=True)

    print('\n'.join(all_projected_filenames))


if __name__ == '__main__':
    main()
