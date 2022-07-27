import glob
import os
import numpy as np
import cv2
cv2.setNumThreads(0) 
import torch
import torch.utils.data as data

class liverDataset(data.Dataset):
    def __init__(self, datasetRoot, setName, usedIndices, isTraining, transform=None):

        self.mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float()
        self.std =  torch.from_numpy(np.array([0.229, 0.224, 0.225])).float()
        self.isTraining = isTraining

        self.imageList = []
        self.labelList = []
        for idx in usedIndices:
            patientImagePath = datasetRoot + '/' + setName + '/' + 'liver_'+ str(idx) + '/'
            patientImageFiles = sorted(glob.glob(patientImagePath + '*.png'))

            patientLabelPath = datasetRoot + '/' + 'labels' + '/' + 'liver_'+ str(idx) + '/'
            patientLabelsFiles = sorted(glob.glob(patientLabelPath + '*.png'))


            self.imageList = self.imageList + patientImageFiles
            self.labelList = self.labelList + patientLabelsFiles

            if len(patientImageFiles) != len(patientLabelsFiles):
                assert('Slice and label not match')

        self.transform = transform
        print('Total slice images', len(self.imageList))

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, idx):

        sliceImageOriginal = cv2.imread(self.imageList[idx])
        sliceLabelOriginal = cv2.imread(self.labelList[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformed = self.transform(image=sliceImageOriginal, mask=sliceLabelOriginal)
            sliceImageOriginal = transformed['image']
            sliceLabelOriginal = transformed['mask']

        
        # Normalize by imagenet mean and std
        sliceImage = cv2.cvtColor(sliceImageOriginal, cv2.COLOR_BGR2RGB)
        sliceImage = sliceImage / 255.0
        sliceImage = torch.from_numpy(sliceImage).float() # swap axis from H,W,C --> C,W,H
        sliceImage = (sliceImage - self.mean) / self.std
        sliceImage = sliceImage.permute(2,0,1)

        
        sliceLabel = sliceLabelOriginal.copy()
        sliceLabel[np.where(sliceLabel==127)] = 1 #==> gray liver to label 1
        sliceLabel[np.where(sliceLabel==255)] = 2 #==> white cancer to label 2
        sliceLabel = torch.from_numpy(sliceLabel).long()
        if self.isTraining:
            return sliceImage, sliceLabel
        else:
            return sliceImage, sliceLabel, sliceImageOriginal, sliceLabelOriginal, self.imageList[idx]


if __name__ == '__main__':
    import albumentations as A
    trainTransform = A.Compose([
        A.Flip(p=0.5),
        #A.RandomBrightnessContrast(p=0.2),
    ])


    o = liverDataset(datasetRoot="dataset/liver", setName="train", isTraining=False,usedIndices=list(range(0,105)), transform=trainTransform)
    sliceImage, sliceLabel, sliceImageOriginal, sliceLabelOriginal, imagePath = o.__getitem__(50)

    print(sliceImage.shape, sliceLabel.shape)

    print(sliceImageOriginal.shape, sliceLabelOriginal.shape, imagePath)
    cv2.imwrite('pic.jpg', sliceImageOriginal)
    cv2.imwrite('lab.jpg', sliceLabelOriginal)