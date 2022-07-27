
import os
import cv2
import numpy as np
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from liverDataset import liverDataset
from utils import AverageMeter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

lr = 1e-3
totalEpoch = 100
testEveryXEpoch = 1
expName = 'resnet18'

outputWeightDirectory ='outputs/'+expName+'/weights/'
outputEvalSampleDirectory ='outputs/'+expName+'/samples/'
os.makedirs(outputEvalSampleDirectory, exist_ok=True)
os.makedirs(outputWeightDirectory, exist_ok=True)

net = smp.FPN(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=3) # use small net as example, without auxilary pspnet head (take ~5min per epoch on our lab server)
net.cuda()
criterion = nn.CrossEntropyLoss().cuda()

liverTrainDataset = liverDataset(datasetRoot="/mnt/datasets/liver", setName="train", isTraining=True, usedIndices=list(range(0,105))) # exclude 105
trainDatasetLoader = DataLoader(liverTrainDataset, batch_size=32, num_workers=16, pin_memory=True, shuffle=True)
liverTestDataset = liverDataset(datasetRoot="/mnt/datasets/liver", setName="train", isTraining=False, usedIndices=list(range(105,131))) # exclude 131
testDatasetLoader = DataLoader(liverTestDataset, batch_size=32, num_workers=16, pin_memory=True, shuffle=True)

optimizer = optim.Adam(net.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()
trainLossAverager = AverageMeter()
bestPixelAccuracy = 0

for currentEpoch in range(totalEpoch):

    trainProgressbar = tqdm(enumerate(trainDatasetLoader),total=len(trainDatasetLoader), dynamic_ncols=True)
    for trainBatchIdx, trainBatchData in trainProgressbar:
        net.train()
        with torch.cuda.amp.autocast(): # Use mixed precision to acclerate training speed
            sliceImages, sliceLabels = trainBatchData
            sliceImages, sliceLabels = sliceImages.cuda(non_blocking=True), sliceLabels.cuda(non_blocking=True)
            trainPreds = net(sliceImages)
            loss = criterion(trainPreds, sliceLabels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        #scheduler.step()
        scaler.update()
        trainLossAverager.update(loss.item())

        currentLR = round(optimizer.param_groups[0]['lr'], 6)
        trainProgressbar.set_description("[{}][Train] avg_loss: {} loss: {} lr:{}".format(currentEpoch+1, round(trainLossAverager.avg,4) ,round(trainLossAverager.val, 4) , currentLR) )

    if currentEpoch % testEveryXEpoch == 0:
        net.eval()
        # Do eval here!
        testPixelAccuracyAverager = AverageMeter()
        testProgressbar = tqdm(enumerate(testDatasetLoader),total=len(testDatasetLoader), dynamic_ncols=True)

        for testBatchIdx, testBatchData in testProgressbar:
            with torch.cuda.amp.autocast(): # Use mixed precision to acclerate training speed
                with torch.no_grad(): # disable grad tracing in eval
                    sliceImagesT, sliceLabelsT, sliceImagesOriginal, sliceLabelsOriginal, imagePaths = testBatchData
                    sliceImagesT, sliceLabelsT = sliceImagesT.cuda(non_blocking=True), sliceLabelsT.cuda(non_blocking=True)
                    testPreds = net(sliceImagesT)

                    predLabels = torch.argmax(testPreds, dim=1)
                    
                    totalPixel = predLabels.nelement()
                    batchPixelAccuracy = torch.sum(torch.eq(predLabels, sliceLabelsT)).item() / totalPixel # pixel acc (you may need to implement your own evaluation method here)
                    testPixelAccuracyAverager.update(batchPixelAccuracy)

                    testProgressbar.set_description("[{}][Test] best_pixelacc:{} cur_pixelacc:{}".format(currentEpoch+1, bestPixelAccuracy, round(testPixelAccuracyAverager.val,5)))

                    if testBatchIdx == 0:
                        predLabelsNP = predLabels.cpu().numpy()
                        predLabelsNP[np.where(predLabelsNP==1)] = 127
                        predLabelsNP[np.where(predLabelsNP==2)] = 255

                        for imageIdx in range(sliceImagesOriginal.shape[0]):
                            sliceImageOriginalNP = sliceImagesOriginal[imageIdx].numpy()

                            sliceLabelOriginalNP = sliceLabelsOriginal[imageIdx].numpy()
                            sliceLabelOriginalNP = np.stack([sliceLabelOriginalNP]*3, axis=2)

                            slicePred = predLabelsNP[imageIdx, :, :]
                            slicePred = np.stack([slicePred]*3, axis=2)

                            outputImage = np.concatenate([sliceImageOriginalNP,sliceLabelOriginalNP, slicePred], axis=1)

                            cv2.imwrite(outputEvalSampleDirectory+str(imageIdx).zfill(2)+'.jpg', outputImage)
                        
        if testPixelAccuracyAverager.avg > bestPixelAccuracy: # best weight save criterion
            bestPixelAccuracy = testPixelAccuracyAverager.avg
            # do something here when new best is found
            torch.save(
            {
                'state_dict':net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acuracy': bestPixelAccuracy,
                'epoch': currentEpoch
            } ,outputWeightDirectory + 'best.pth')

        outputString = '[{}][Test] best_pixelacc:{} cur_pixelacc:{}'.format(currentEpoch+1, bestPixelAccuracy,round(testPixelAccuracyAverager.avg,5))
        print(outputString)

        # save current epoch weight
        torch.save(
            {
                'state_dict':net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acuracy': bestPixelAccuracy,
                'epoch': currentEpoch
            } ,outputWeightDirectory + str(currentEpoch) +'.pth')

        
        

