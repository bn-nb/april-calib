import os
import cv2
import numpy as np
from calibrate import ARUCO


def getTvecDist(tvec):
    dist = 0
    for i in tvec:
        dist += i**2
    return dist ** 0.5


def inversePerspective(rvec, tvec):
    # https://stackoverflow.com/a/52188699
    R, _ = cv2.Rodrigues(rvec)
    Rinv = R.T
    invTvec = np.dot(Rinv, -tvec)
    invRvec, _ = cv2.Rodrigues(Rinv)
    return invRvec, invTvec


def getCamPoseFromNearestTag(img,family,mSize,camMat,dstCof):
    
    if (tuppy := ARUCO.detectAPRILTags(img, family)) is not None:

        basic = np.array(((-1,1,0),(1,1,0),(1,-1,0),(-1,-1,0)),dtype=np.float32)
        objpt = basic * (mSize/2)
        
        corners, ids, numtags = tuppy

        params = {
            'objectPoints': objpt,
            'imagePoints': None,
            'cameraMatrix': camMat,
            'distCoeffs': dstCof,
            'flags': cv2.SOLVEPNP_IPPE_SQUARE,
        }

        min_rvec = np.zeros((3,1))
        min_tvec = np.full((3,1), np.inf)

        for i in range(numtags):
            params['imagePoints'] = corners[i]
            _,r,t = cv2.solvePnP(**params)

            if getTvecDist(min_tvec) > getTvecDist(t):
                min_tvec = t
                min_rvec = r

        invR, invT = inversePerspective(min_rvec, min_tvec)

        invR = tuple(invR.ravel() * 180 / np.pi)
        invT = tuple(invT.ravel())

        return invR, invT



if __name__ == "__main__":

    vid_i = cv2.VideoCapture(0)
    vid_i.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid_i.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    CAM_MAT = np.load(os.path.join(ARUCO.ROOT_DIR, 'cameraMatrix.npy'))
    DST_COF = np.load(os.path.join(ARUCO.ROOT_DIR, 'distorCoeffs.npy'))
    IMG_DIR = os.path.join(ARUCO.ROOT_DIR, 'calibtest')
    IMG_LST = [os.path.join(IMG_DIR,x) for x in os.listdir(IMG_DIR)]

    params = {
        'img': None,
        'family': ARUCO.APRIL_16H5,
        'mSize': 6.75, # cms
        'camMat': CAM_MAT,
        'dstCof': DST_COF,
    }

    while (read_i := vid_i.read())[0]:

        params['img'] = read_i[1]
        ARUCO.drawMarkers(read_i[1], [params['family']])
        camPose = getCamPoseFromNearestTag(**params)

        window1 = "TEST"
        cv2.namedWindow(window1)
        cv2.moveWindow(window1, 100, 50)
        cv2.imshow(window1, read_i[1])
        key = chr(cv2.waitKey(1) % 256)

        if   key in 'qQ': break
        elif key in 'dD': print(camPose, sep='\n\n')

    vid_i.release()
    cv2.destroyAllWindows()
