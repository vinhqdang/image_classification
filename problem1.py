import os
import sys
import argparse
import errno
from scipy import misc
from operator import mul
from collections import namedtuple
import matplotlib.pyplot as plt

# define some constants
MAX_PIXEL_VALUE=255
MIN_PIXEL_VALUE=0

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Finding max rectangle in a binary image that contains only ones.")

    parser.add_argument('--input', nargs='?', default='binary_images/binary_image_1.png',
                        help='Input image path')

    return parser.parse_args()

def test_1 ():
    """
    First test function.
    Use with py.test
    """
    assert (find_max_rectangle("binary_images/binary_image_1.png") == (71,120,135,170))

def test_max_histogram_1 ():
    assert max_rectangle_area([2,6,0,6,5,1])[0] == 10

def test_max_histogram_2 ():
    assert max_rectangle_area([1,5,3,5,4,0])[0] == 12

def test_max_histogram_3 ():
    assert max_rectangle_area([1,5,3,5,4,0])[1] == 5 

def test_max_histogram_4 ():
    assert max_rectangle_area([1,5,3,5,4,0])[2] == 1

def test_max_histogram_5 ():
    assert max_rectangle_area([1,5,3,5,4,0])[3] == 3

def test_max_matrix_1 ():
    mat = [[1,0,0,1,1,1],[1,0,1,1,0,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]
    assert max_size (mat, value=1)[0] == 2

def test_max_matrix_2 ():
    mat = [[1,0,0,1,1,1],[1,0,1,1,0,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]
    assert max_size (mat, value=1)[1] == 2

def test_max_matrix_3 ():
    mat = [[1,0,0,1,1,1],[1,0,1,1,0,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]
    assert max_size (mat, value=1)[2] == 5

def test_max_matrix_4 ():
    mat = [[1,0,0,1,1,1],[1,0,1,1,0,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]
    assert max_size (mat, value=1)[3] == 3

Info = namedtuple('Info', 'start height')

def max_size(mat, value=1):
    """
    Find the coordinates of largest area in the matrix which contains only values.

    Args:
        mat: the binary input matrix
        value: 0 or 1

    Returns:
        a coordinates as a tuple (xmin, ymin, xmax, ymax)
    """
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_size, max_pos, max_start, max_height = max_rectangle_area(hist)
    max_row = 0
    row_cnt = 1
    for row in it:
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        cur_max_size, cur_max_pos, cur_max_start, cur_max_height = max_rectangle_area(hist)
        if cur_max_size > max_size:
            max_size = cur_max_size
            max_pos = cur_max_pos
            max_start = cur_max_start
            max_height = cur_max_height
            max_row = row_cnt
        row_cnt += 1
    # print (max_size, max_pos, max_start, max_height, max_row)

    xmin = max_start
    ymin = max_row - max_height + 1
    
    xmax = max_pos - 1
    ymax = max_row
    
    return (xmin, ymin, xmax, ymax)

def max_rectangle_area(histogram):
    """
    Find the area of the largest rectangle that fits entirely under
    the histogram.

    Args:
        histogram: the histogram as an array

    Returns:
        a tuple (max_area, max_pos, max_start, max_height) where
        max_area = (max_pos - max_start) * max_height
    """
    stack = []
    top = lambda: stack[-1]
    max_area = 0
    pos = 0 # current position in the histogram
    
    # store the information of largest area so far
    max_pos = 0
    max_height = 0
    max_start = 0

    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                cur_area = top().height*(pos-top().start)
                if cur_area > max_area:
                    max_area = cur_area
                    max_pos = pos
                    max_start = top().start
                    max_height = top().height                
                start, _ = stack.pop()
                continue
            break # height == top().height goes here

    pos += 1
    for start, height in stack:
        cur_area = height*(pos-start)
        if cur_area > max_area:
            max_area = cur_area
            max_pos = pos
            max_height = height
            max_start = start
    return (max_area, max_pos, max_start, max_height)

def find_max_rectangle (img_path):
    """
    Finding a maximum rectangle in a binary image that contains only ones

    Args:
        img_path: the image file path.

    Returns:
        returns a tuple (xmin, ymin, xmax, ymax) that represents the max rectangle.

    Raises:
        IOError: file does not exist.
        SyntaxError: input file is not a binary image
    """
    if not os.path.isfile(img_path):
        raise IOError("The input file does not exist.")
    try:
        pic = misc.imread(img_path)
    except Exception, e:
        raise e

    # create a binary matrix from the input binary image
    bin_matrix = []
    for i in range (pic.shape[0]):
        cur_row = []
        for j in range (pic.shape[1]):
            pixel = pic[i,j]
            if pixel[0] == pixel[1] == pixel[2] == MAX_PIXEL_VALUE:
                cur_row.append (1)
            elif pixel[0] == pixel[1] == pixel[2] == MIN_PIXEL_VALUE:
                cur_row.append (0)
            else:
                raise SyntaxError("The input file is not a binary image.")
        if i > 0 and len(cur_row) != len (bin_matrix[0]):
            raise SyntaxError("The length of rows are different.")
        bin_matrix.append(cur_row)
    
    return max_size(bin_matrix, value=1)

if __name__ == "__main__":
    args = parse_args()
    # print (max_rectangle_area([2,6,0,6,5,1]))
    # print (max_rectangle_area([1,5,3,5,4,0]))
    # mat = [[1,0,0,1,1,1],[1,0,1,1,0,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]
    # print (max_size(mat,1))
    try:
        print (find_max_rectangle(args.input))
    except Exception, e:
        print (e)
    