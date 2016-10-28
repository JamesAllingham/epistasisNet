"""This script is for converting between the data format of GAMETES and MDR to BEAM.

This is used so that comparisons can be made with BEAM in addition to MDR.
"""

import getopt
import sys

import numpy as np


def main(args):
    """The main function which executes all of the script functionality.
    """

    error_string = 'test.py -i <input file> -o <output file>'

    input_file_name_and_path = ''
    output_file_name_and_path = ''

    try:
        opts, args = getopt.getopt(args, "hi:o:", ["infile=", "outfile="])
    except getopt.GetoptError:
        print(error_string)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(error_string)
            sys.exit(2)
        elif opt in ("-i", "--infile"):
            input_file_name_and_path = arg
        elif opt in ("-o", "--outfile"):
            output_file_name_and_path = arg

    print('Input file is %s'% input_file_name_and_path)
    print('Output file is %s'% output_file_name_and_path)

    # Read the input data into an np array
    try:
        input_data = np.genfromtxt(input_file_name_and_path, dtype='intc')
    except IOError:
        print('Unable to read the input file.')
        print('Script usage:')
        print(error_string)
        sys.exit(2)

    # Rotate the data so that the last collumn is the first row
    rotated = np.rot90(input_data, 3)
    header = " ".join([str(elem) for elem in range(np.shape(rotated)[1])])

    try:
        np.savetxt(output_file_name_and_path, rotated, fmt='%i', delimiter=' ', header=header, comments='')
    except IOError as excep:
        print(excep)
        print('Unable to write the output file.')
        print('Script usage:')
        print(error_string)
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
