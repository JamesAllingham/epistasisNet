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
        input_data = np.genfromtxt(input_file_name_and_path, dtype='intc', skip_header=1)
    except IOError:
        print('Unable to read the input file.')
        print('Script usage:')
        print(error_string)
        sys.exit(2)

    # Rotate the data so that the last collumn is the first row
    rotated = np.rot90(input_data)
    num_rows, num_cols = rotated.shape

    result = np.zeros((num_rows, num_cols*2))

    # For the case control row every collumn is duplicated
    for j in range(num_cols):
        result[0, 2*j] = rotated[0, j]
        result[0, 2*j+1] = rotated[0, j]

    # For the other rows:
    # 2 -> 1,1; 1 -> 1,0; 0 -> 0,0
    for i in xrange(1, num_rows):
        for j in range(num_cols):
            if rotated[i, j] == 1:
                result[i, 2*j] = 1
            elif rotated[i, j] == 2:
                result[i, 2*j+1] = 1
                result[i, 2*j] = 1
            elif rotated[i, j] != 0:
                print('An allele value other than 0, 1, or 2 is invalid.')
                sys.exit(2)

    try:
        np.savetxt(output_file_name_and_path, result, fmt='%i', delimiter=' ')
    except IOError as excep:
        print(excep)
        print('Unable to write the output file.')
        print('Script usage:')
        print(error_string)
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
