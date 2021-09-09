import pandas as pd
import nibabel as nib
import numpy as np
import time
import cv2 as cv
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2

np.random.seed(0)

def Rx(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = 1., 0. , 0.
    r21, r22, r23 = 0., np.cos(x), -np.sin(x)
    r31, r32, r33 = 0., np.sin(x), np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Ry(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), 0., np.sin(x)
    r21, r22, r23 = 0., 1., 0.
    r31, r32, r33 = -np.sin(x), 0, np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Rz(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def rotate_plane(plane, rotationMatrix):
	return np.matmul(rotationMatrix, plane)

def normalize(img):
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img = 255 * img 
    return np.uint8(img)

if __name__ == "__main__":
    

    writeDir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/train-val2/"
    readDir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/normals/train-val2/"
    trainScans = ["SMIR.Body.021Y.M.CT.57761", "SMIR.Body.025Y.M.CT.59477", "SMIR.Body.030Y.F.CT.59466", 
                "SMIR.Body.030Y.F.CT.59471", "SMIR.Body.033Y.M.CT.57764", "SMIR.Body.036Y.F.CT.58319", 
                "SMIR.Body.037Y.M.CT.57613", "SMIR.Body.037Y.M.CT.59473", "SMIR.Body.041Y.F.CT.57699", 
                "SMIR.Body.043Y.M.CT.58317", "SMIR.Body.045Y.M.CT.59467", "SMIR.Body.045Y.M.CT.59476", 
                "SMIR.Body.045Y.M.CT.59481", "SMIR.Body.047Y.F.CT.57792", "SMIR.Body.049Y.M.CT.59482", 
                "SMIR.Body.052Y.M.CT.57765", "SMIR.Body.052Y.M.CT.59475",  "SMIR.Body.057Y.F.CT.57793", 
                 "SMIR.Body.057Y.M.CT.59483", "SMIR.Body.058Y.M.CT.57767"]

    testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", 
            "SMIR.Body.040Y.M.CT.57768", "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", 
            "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]

    # # Load ct volume
    # ctID = trainScans[0]

    # #imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii" # training ct scan
    # imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody/{}/{}.nii".format(ctID, ctID)
    # N = 512 
    # print("INFO: loading CT volume...")
    # tic_load = time.time()
    # ctVolume = nib.load(imgpath).get_fdata().astype(int)
    # ctVolume = np.squeeze(ctVolume)
    # voi = ctVolume[:,:, :N] # Extracts volume of interest (512 x 512 x 512) from the full body ct volume (512 x 512 x 3000)
    # #voi = ctVolume[:,:, -N:]
    # #voi = voi[..., ::-1] # inverses slices order as image appears upside down
    # voi = voi - np.min(voi) # shift to avoid negative CT numbers
    # voi = np.pad(voi, N//2, "constant", constant_values=0)
    # toc_load = time.time()
    # print("Done after {:.2f} seconds.".format(toc_load - tic_load)) 

    # tic_fft = time.time()
    # voiShifted = fftshift(voi)
    # del voi
    # voiFFT = fftn(voiShifted)
    # del voiShifted
    # voiFFTShifted = fftshift(voiFFT)
    # del voiFFT
    # toc_fft = time.time()
    # print("3D FFT computed in {:.2f} seconds.".format(toc_fft - tic_fft))


    # # Rotation and Interpolation of the projection slice from the 3D FFT volume

    # x = np.linspace(-N+0.5, N-0.5, 2*N)
    # y = np.linspace(-N+0.5, N-0.5, 2*N)
    # z = np.linspace(-N+0.5, N-0.5, 2*N)


    # projectionPlane = np.array([[xi, 0, zi] for xi in x for zi in z])
    # projectionPlane = np.reshape(projectionPlane, (2*N, 2*N, 3, 1), order="F")

    # def render_view(viewpoint):
    #     theta_x = viewpoint[0] #np.random.randint(-10, 10)
    #     theta_y = viewpoint[1] #np.random.randint(-10, 10)
    #     theta_z = viewpoint[2]
    #     #tx = viewpoint[1]
    #     #ty = viewpoint[2]
    #     tic_rendering = time.time()
    #     rotationMatrix = Rx(theta_x) @ Rz(theta_z) # @ Ry(theta_y)
    #     projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
    #     projectionSliceFFT = interpn(points=(x, y, z), values=voiFFTShifted, xi=projectionSlice, method="linear",
    #                                  bounds_error=False, fill_value=0)      
    #     img = np.abs(fftshift(ifft2(projectionSliceFFT)))
    #     img = img[N//2:N+N//2, N//2:N+N//2]
    #     img = normalize(img)
    #     #img = img[54+tx:454+tx, 63+ty:463+ty]
    #     #img = cv.resize(img, (400, 400), interpolation=cv.INTER_AREA)
    #     #img = img[56+tx:456+tx, 56+ty:456+ty]
    #     cv.imwrite("./{}_{}_{}.png".format(ctID, theta_x, theta_z), img)
    #     toc_rendering = time.time()
    #     print("viewpoint = {} {} {}\t {:.2f} seconds".format(theta_x, theta_y, theta_z, toc_rendering-tic_rendering))
        
    elevation = [ i for i in range(0, 180, 10)]
    inplane =  [i for i in range(0, 360, 10)]
    azimuth = [i for i in range(0, 360, 10)]
    df = pd.DataFrame()
    labels_x = []
    labels_z = []
    image_name = []

    viewpoints = [(x, 0, z) for  x in elevation for z in azimuth]
    for ctID in testScans:
        for viewpoint in viewpoints:
            #render_view(viewpoint)
            labels_x.append(viewpoint[0])
            labels_z.append(viewpoint[2])
            image_name.append("{}_{}_{}.png".format(ctID, viewpoint[0], viewpoint[2]))
    df["image"] = image_name
    df["elevation"] = labels_x
    df["azimuth"] = labels_z
    df.to_csv("test.csv", sep=",", index=False)
