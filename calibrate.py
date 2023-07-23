import os
import cv2
import numpy as np
from datetime import datetime as dt


class ARUCO:
    # Calibration options - 100 * size_in_cms
    # Do not create instances of this class

    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    @staticmethod
    def __getBoard(boardprops, layout):
        # 100 * size_in_cms
        board = cv2.aruco.GridBoard_create(*boardprops[:-1])
        outimg = board.draw(layout, marginSize=boardprops[-1])
        # cv2.imshow("TEST", outimg)
        return board

    @staticmethod
    def detectAPRILTags(img, family, color=True):

        if color:  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejections = cv2.aruco.detectMarkers(
            img, family, parameters=ARUCO.PARAMS_CMN)
        num_tags = len(corners)

        if (num_tags):
            return corners, ids, num_tags
        else:
            return None

    @staticmethod
    def drawMarkers(img, list_of_aruco_families, *, attest=True, debug=None):

        for tagFamily in list_of_aruco_families:
            temp = ARUCO.detectAPRILTags(img, tagFamily)

            if (temp != None):
                T = temp[1] if attest else None
                cv2.aruco.drawDetectedMarkers(img, temp[0], T)
                
                if isinstance(debug, list):
                    debug.append(temp)

    APRIL_25H9 = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_25H9)
    APRIL_16H5 = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16H5)
    APRIL_FAMS = (APRIL_25H9, APRIL_16H5)
    PARAMS_CMN = cv2.aruco.DetectorParameters_create()


    # GRID_16H5 = __getBoard(
    #     boardprops=(5, 6, 260, 130, APRIL_16H5, 150),
    #     layout=(2100, 2970)
    #     )

    # GRID_25H9 = __getBoard(
    #     boardprops=(5, 7, 270, 120, APRIL_25H9, 170),
    #     layout=(2100, 2970)
    #     )

    ZERO_16H5 = __getBoard(
        boardprops=(1, 1, 675, 100, APRIL_16H5, 0, 110),
        layout=(900, 1050)
        )

    # FIVE_25H9 = __getBoard(
    #     boardprops=(1, 1, 700, 100, APRIL_25H9, 5, 110),
    #     layout=(900, 1050)
    #     )


def collectSamples(N, imgs_dir, cameraPath=0):
    # Pass absolute path preferably
    print("Collecting:", N, "samples...")
    vid_i = cv2.VideoCapture(cameraPath)
    vid_i.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid_i.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    count = 0

    while (read_i := vid_i.read())[0]:
        debug, outimg = [], read_i[1].copy()
        ARUCO.drawMarkers(read_i[1], ARUCO.APRIL_FAMS, debug=debug)
        
        window1 = "TEST"
        cv2.namedWindow(window1)
        # cv2.moveWindow(window1, 100, 50)
        cv2.imshow(window1, read_i[1])
        key = chr(cv2.waitKey(1) % 256)

        if   key in 'qQ': break
        elif key in 'dD': print(debug)
        elif key in 'cC':
            if (len(debug)):
                path = str(dt.now()).replace(':', '') + '.jpg'
                path = os.path.join(imgs_dir, path)
                cv2.imwrite(path, outimg)
                count += 1
                if (count == N):
                    print(N, 'samples collected!\n')
                    break

    vid_i.release()
    cv2.destroyAllWindows()


def getImages_Non_Recursive(imgs_dir):
    # Pass absolute path as argument preferably
    IMG_EXTS = ('.jpg', '.png', '.jpeg')
    for node in os.listdir(imgs_dir):
        if os.path.isfile(path := os.path.join(imgs_dir, node)):
            if path.lower().endswith(IMG_EXTS):
                yield path


def getCameraMatrix(
    root_dir, imgAbsPaths, family, 
    board, camInit=None, distCoeffs=None
    ):

    # corners,ids must be combined lists of corners, ids
    # detected from all images, in order of appearance
    # the counter param is used to iterate C,I per image

    sample_img = cv2.imread(imgAbsPaths[0])
    C,I,N = ARUCO.detectAPRILTags(sample_img, family)
    H,W,_ = sample_img.shape

    calib_params = {
        'corners': C, 
        'ids': I, 
        'counter': [N], 
        'board': board,
        'imageSize': (H,W),
        'cameraMatrix': camInit, 
        'distCoeffs': distCoeffs,
        'flags': 0 + cv2.CALIB_USE_INTRINSIC_GUESS * (camInit is not None),
    }

    for img in imgAbsPaths[1:]:
        tmp = ARUCO.detectAPRILTags(cv2.imread(img), family)
        if (tmp != None):
            calib_params['corners'] = np.vstack([calib_params['corners'], tmp[0]])
            calib_params['ids'] = np.vstack([calib_params['ids'], tmp[1]])
            calib_params['counter'].append(tmp[2])

    calib_params['counter'] = np.array(calib_params['counter'])
    tuppy = cv2.aruco.calibrateCameraAruco(**calib_params)
    rms_reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs = tuppy

    camMatPath = os.path.join(root_dir, 'cameraMatrix.npy')
    dstCofPath = os.path.join(root_dir, 'distorCoeffs.npy')
    np.save(camMatPath, cameraMatrix)
    np.save(dstCofPath, distCoeffs)

    return camMatPath, dstCofPath


if __name__ == "__main__":
    # Unit for size = 10e-5 m
    # 100 Pixels = 1 cm for saving images only
    # Prefer passing absolute paths

    IMGS_DIR = os.path.join(ARUCO.ROOT_DIR, 'calibtest')

    try:                    os.mkdir(IMGS_DIR)
    except FileExistsError: pass
    except Exception as e:  print(e)

    params = {
        'root_dir': ARUCO.ROOT_DIR,
        'imgAbsPaths': None,
        'family':ARUCO.APRIL_16H5,
        'board':ARUCO.ZERO_16H5,
    }
    
    c = np.array([
        [1000, 0, 700],
        [0, 1000, 400],
        [0, 0, 1]])

    d = np.zeros((5,1))

    cameraPath = 0
    params['imgAbsPaths'] = collectSamples(60, IMGS_DIR, cameraPath)
    params['imgAbsPaths'] = list(getImages_Non_Recursive(IMGS_DIR))

    for i in range(int(input("Enter number of iterations: "))):
        params['camInit'] = c
        params['distCoeffs'] = d
        c,d = getCameraMatrix(**params)
        c,d = np.load(c), np.load(d)
        print(c,d,'\n',sep='\n')
