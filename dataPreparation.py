import os
import glob
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def extractLiverImagesAndLabels(outputDatasetPath, niiFilePath):
    fileNameWithoutExtension = os.path.basename(niiFilePath).split('.')[0]
    #print(niiFilePath)

    if 'imagesTr' in niiFilePath: # train set
        currentPatientOutputDir = outputDatasetPath+'/train/'+fileNameWithoutExtension+'/'
        
    elif 'imagesTs' in niiFilePath: # test set
        currentPatientOutputDir = outputDatasetPath+'/test/'+fileNameWithoutExtension+'/'
    
    elif 'labels' in niiFilePath:
        currentPatientOutputDir = outputDatasetPath+'/labels/'+fileNameWithoutExtension+'/'
    else: # this should not happen
        assert("Invalid path?")

    os.makedirs(currentPatientOutputDir, exist_ok=True)

    if 'imagesTr' in niiFilePath or 'imagesTs' in niiFilePath:
        img = nib.load(niiFilePath)
        imgNP = img.get_fdata() # H, W, Channel
        for sliceIdx in range(imgNP.shape[2]):
            sliceImage = imgNP[:,:, sliceIdx]
            sliceNorm = sliceImage - sliceImage.min()
            sliceNorm = sliceNorm / sliceNorm.max()
            sliceNorm = (sliceNorm * 255).astype(np.uint8)

            cv2.imwrite(currentPatientOutputDir+str(sliceIdx).zfill(3)+'.png', sliceNorm, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    elif 'labels' in niiFilePath:
        labels = nib.load(niiFilePath)
        labelsNP = labels.get_fdata() # H, W, Channel

        # output array in H, W, sliceNo
        #"labels": { BGR
        #    "0": "background", 0
        #    "1": "liver", 127
        #    "2": "cancer" 255
        #}
        labelsNP = labelsNP.astype(np.uint8)

        for sliceIdx in range(labelsNP.shape[2]):
            currentSliceLabel = labelsNP[:,:,sliceIdx]
            currentSliceLabel[np.where(currentSliceLabel==1)] = 127 
            currentSliceLabel[np.where(currentSliceLabel==2)] = 255
            cv2.imwrite(currentPatientOutputDir+str(sliceIdx).zfill(3)+'.png', currentSliceLabel, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            readBackIm = cv2.imread(currentPatientOutputDir+str(sliceIdx).zfill(3)+'.png', cv2.IMREAD_GRAYSCALE)
            np.allclose(currentSliceLabel, readBackIm)
        
    print(niiFilePath)

if __name__ == '__main__':
    outputDatasetPath = 'dataset/liver/'
    niiFileTrainPaths = sorted(glob.glob('dataset/Task03_Liver/imagesTr/**/*.nii.gz', recursive=True))
    niiFileTestPaths = sorted(glob.glob('dataset/Task03_Liver/imagesTs/**/*.nii.gz', recursive=True))
    niiFileLabelPaths = sorted(glob.glob('dataset/Task03_Liver/labelsTr/**/*.nii.gz', recursive=True))
    niiFilePaths = niiFileTrainPaths + niiFileTestPaths + niiFileLabelPaths
    #niiFilePaths = niiFileLabelPaths
    

    partialFuncCall = partial(extractLiverImagesAndLabels, outputDatasetPath)
    with ProcessPoolExecutor(max_workers=16) as executor: # make it multiprocess (carefully adjust this to depends on your RAM!!)
        executor.map(partialFuncCall, niiFilePaths)

    #for niiFilePath in niiFilePaths:
    #    extractLiverImagesAndLabels(outputDatasetPath, niiFilePath)