import numpy as np
import math
import argparse
import cv2
from scipy import fftpack
from utils import *
import huffman

def block_to_zigzag(block):
    # *: unpacking arguments list
    # print(*block.shape)
    return np.array([block[point] for point in zigzag_points(*block.shape)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest='input', default='rmb.bmp', help="path to the input image")
    args = parser.parse_args()

    input_file = args.input

    with open(input_file, "rb") as binary_file:
        data = binary_file.read()
        print(len(data))
        print(data[100])
    exit()
    
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)

    original_rows, original_cols = img.shape[0], img.shape[1]

    rows = original_rows
    cols = original_cols
    padding_rows = 0
    padding_cols = 0
    if original_rows % 8 != 0:
        for i in range(1,9):
            rows = original_rows + i
            if rows % 8 == 0:
                padding_rows = i
                break
    if original_cols % 8 != 0:
        for i in range(1,9):
            cols = original_cols + i
            if cols % 8 == 0:
                padding_cols = i
                break

    pad_top = math.floor(padding_rows / 2.0)
    pad_bot = math.ceil(padding_rows / 2.0)
    pad_left = math.floor(padding_cols / 2.0)
    pad_right = math.ceil(padding_cols / 2.0)
    # print(pad_top, pad_bot, pad_left, pad_right)

    # Padding to multiples of 8
    img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REPLICATE)

    img_trans = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    B = 8
    # Downsampling
    SSV = 2
    SSH = 2
    crf = cv2.boxFilter(img_trans[:,:,1], ddepth=-1, ksize=(2,2))
    cbf = cv2.boxFilter(img_trans[:,:,2], ddepth=-1, ksize=(2,2))
    crsub = crf[::SSV, ::SSH]
    cbsub = cbf[::SSV, ::SSH]
    imSub = [img_trans[:,:,0], crsub, cbsub]

    Q = load_quantization_table()

    dcs = []
    acs = []
    for idx, channel in enumerate(imSub):
        channelrows = channel.shape[0]
        channelcols = channel.shape[1]
        blocksV = channelrows // B
        blocksH = channelcols // B
        vis0 = np.zeros((channelrows, channelcols), np.float)
        vis0[:channelrows, :channelcols] = channel
        vis0 = vis0 - 128
        # dc is the top-left cell of the block, ac are all the other cells
        dc = np.empty((blocksV, blocksH), dtype=np.int32)
        ac = np.empty((blocksV, blocksH, 63), dtype=np.int32)
        for row in range(blocksV):
            for col in range(blocksH):
                currentblock = cv2.dct(vis0[row*B:(row+1)*B, col*B:(col+1)*B])
                quant_block = np.round(currentblock / Q[idx])

                zz = block_to_zigzag(quant_block)
                dc[row, col] = zz[0]
                ac[row, col, :] = zz[1:]
        dcs.append(dc)
        acs.append(ac)



    
    cv2.imshow('rmb_y', imSub[0])
    # cv2.imshow('rmb', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
