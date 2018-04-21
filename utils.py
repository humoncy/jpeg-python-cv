import numpy as np
import cv2


def load_quantization_table():
    QY = np.array([[16,11,10,16,24,40,51,61],
                   [12,12,14,19,26,48,60,55],
                   [14,13,16,24,40,57,69,56],
                   [14,17,22,29,51,87,80,62],
                   [18,22,37,56,68,109,103,77],
                   [24,35,55,64,81,104,113,92],
                   [49,64,78,87,103,121,120,101],
                   [72,92,95,98,112,100,103,99]])

    QC = np.array([[17,18,24,47,99,99,99,99],
                   [18,21,26,66,99,99,99,99],
                   [24,26,56,99,99,99,99,99],
                   [47,66,99,99,99,99,99,99],
                   [99,99,99,99,99,99,99,99],
                   [99,99,99,99,99,99,99,99],
                   [99,99,99,99,99,99,99,99],
                   [99,99,99,99,99,99,99,99]])

    # QF = 99
    # if QF < 50 and QF >= 1:
    #     scale = np.floor(5000/QF)
    # elif QF < 100:
    #     scale = 200 - 2*QF
    # else:
    #     print("Quality Factor must be in the range [1..99]")

    # scale = scale / 100.0
    # Q = [QY * scale, QC * scale, QC * scale]

    Q = [QY, QC, QC]

    return Q

def zigzag_points(rows, cols):
    # constants for directions
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # move the point in different directions
    def move(direction, point):
        return {
            UP: lambda point: (point[0] - 1, point[1]),
            DOWN: lambda point: (point[0] + 1, point[1]),
            LEFT: lambda point: (point[0], point[1] - 1),
            RIGHT: lambda point: (point[0], point[1] + 1),
            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
        }[direction](point)

    # return true if point is inside the block bounds
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # start in the top-left cell
    point = (0, 0)

    # True when moving up-right, False when moving down-left
    move_up = True

    for i in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)

def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result

def crop_image(img):
    new_img = np.empty((40, 40))
    new_img = img[390:430, 250:290]

    return new_img

if __name__ == '__main__':
    input_file = "rmb.bmp"
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    small_img = crop_image(img)
    # cv2.imshow("small", small_img)
    # cv2.waitKey(0)
    cv2.imwrite("small_rmb.bmp", small_img)
