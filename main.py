import torch
import os.path
from colorama import init, Fore, Style
import pandas as pd
from Modules.MLP import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle


class Main():

    path = None
    iteration = 1
    validationPercent = testPercent = 1
    learningRate = 0.1
    batchSize = 20
    optimizerType = None
    trainLoader = valLoader = testLoader = None
    hiddenSize1 = 128
    hiddenSize2 = 64
    hiddenSize3 = 32

    scaler = MinMaxScaler(feature_range=(0,1))


    def colorText(self, text, color):
        init()
        colorCode = ""
        if color == "G":
            colorCode = "\033[32m"
        else:
            colorCode = "\033[31m"
        return f"{colorCode}{text}\033[0m"
    

    def checkGPU(self):
        global device
        if torch.cuda.is_available():
            print("CUDA is available")
            numberOfGpus = torch.cuda.device_count()
            print(f"Number of available GPUs: {numberOfGpus}")
            for i in range (numberOfGpus):
                gpuProperties = torch.cuda.get_device_properties(i)
                print(f"GPU{i}: {gpuProperties.name}, (CUDA cores: {gpuProperties.multi_processor_count})")
                device = torch.device("cuda")
            return True
        else:
            print("OOps! your GPU doesn't support required CUDA version.")
            return False
        

    def getDatasetPath(self):
        global path
        path = input("Where can i find the dataset?(Write the path to dataset):  ")
        if os.path.isfile(path + '/train.csv'):
            print(self.colorText("Train dataset exist", "G"))
            return True
        else:
            print(self.colorText("Dataset doesn't exist. Check the directory!", "R"))
            return False
    

    def getUserParams(self):
        global iteration, validationPercent, learningRate, testPercent, batchSize, optimizerType, hiddenSize1, hiddenSize2, hiddenSize3
        iteration = int(input("Enter iteration number: "))
        validationPercent = int(input("Enter validation percent: %"))/100
        testPercent = int(input("Enter test percent: %"))/100
        learningRate = float(input("Enter learning rate: "))
        batchSize = int(input("Enter batch size:(default 20): "))
        optimizerType = input("Which optimizer do you want to choose?(SGD/Adam): ")
        hiddenSize1 = int(input("Enter number of neurons in first layer:(default 128): "))
        hiddenSize2 = int(input("Enter number of neurons in second layer:(default 64): "))
        hiddenSize3 = int(input("Enter number of neurons in third layer:(default 32): "))


    
    def loadDataFromCsv(self):
        global trainLoader, valLoader, testLoader
        dataset = pd.read_csv(path + '/train.csv')
        print("A quick peek o dataset! ...\n")
        print(dataset.head())
        x = dataset.iloc[:, 1:785]
        y = dataset.iloc[:, 0]
        print("Generating train, validation and test sets ...\n")
        xTrainTemp, xValTemp, yTrain, yVal = train_test_split(x, y, test_size=validationPercent)
        xTrainTemp, xTestTemp, yTrain, yTest = train_test_split(xTrainTemp, yTrain, test_size=testPercent)
        print("Scaling data ...\n")
        xTrain = self.scaler.fit_transform(xTrainTemp)
        xVal = self.scaler.transform(xValTemp)
        xTest = self.scaler.transform(xTestTemp)
        print("Generationg Dataloader ...\n")
        tensorXTrain = torch.Tensor(xTrain)
        tensorYTrain = torch.Tensor(yTrain.to_numpy())#.unsqueeze(1)
        tensorXVal = torch.Tensor(xVal)
        tensorYVal = torch.Tensor(yVal.to_numpy())#.unsqueeze(1)
        tensorXTest = torch.Tensor(xTest)
        tensorYTest = torch.Tensor(yTest.to_numpy())#.unsqueeze(1)
        trainDataset = CustomDataset(tensorXTrain, tensorYTrain)
        valDataset = CustomDataset(tensorXVal, tensorYVal)
        testDataset = CustomDataset(tensorXTest, tensorYTest)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)
        testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    

    def saveModel(self, model):
        filename = 'MLP-Digitrecognize.sav'
        pickle.dump(model, open(filename, 'wb'))



    def startNN(self):
        if True:
            if self.getDatasetPath():
                self.getUserParams()
                self.loadDataFromCsv()
                inputDim = 784
                outputDim = 10
                model = MLP(inputDim, hiddenSize1, hiddenSize2, hiddenSize3, outputDim)#.to("cuda")
                lossFunc = nn.CrossEntropyLoss()
                if optimizerType == "SGD":
                    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
                training_losses = []
                print(f"Model is training on {iteration} of epochs")
                print(f"Model validation percent: %{validationPercent*100}")
                print(f"Model test percent: %{testPercent*100}")
                print(f"Model learning rate: {learningRate}")
                print(f"Model optimizer: {optimizerType}")
                print(f"Model batch size: {batchSize}")
                print(f"Model first layer neurons: {hiddenSize1}")
                print(f"Model second layer neurons: {hiddenSize2}")
                print(f"Model third layer neurons: {hiddenSize3}")
                for epoch in range(iteration):
                    for inputs, labels in trainLoader:
                        labels = labels.type(torch.LongTensor)
                        #inputs, labels = inputs.to("cuda"), labels.to("cuda")
                        #labels = labels.type(torch.LongTensor)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = lossFunc(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    training_losses.append(loss.item())
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in valLoader:
                        #inputs, labels = inputs.to("cuda"), labels.to("cuda")
                        outputs = model(inputs)
                        _, predicted_classes = torch.max(outputs, dim=1)  # Get the class with highest probability
                        correct += (predicted_classes == labels).sum().item()
                        total += labels.size(0)
                    accuracy = correct / total
                    print(f"Validation Accuracy: {accuracy:.2f}")
                model.train()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in testLoader:
                        labels = labels.type(torch.LongTensor)
                        #inputs, labels = inputs.to("cuda"), labels.to("cuda")
                        #labels = labels.type(torch.LongTensor)
                        outputs = model(inputs)
                        _, predicted_classes = torch.max(outputs, dim=1)  # Get the class with highest probability
                        correct += (predicted_classes == labels).sum().item()
                        total += labels.size(0)
                        accuracy = correct / total
                        print(f"Test Accuracy: {accuracy:.2f}")
                if input("Do you want to save the model?(y/n): ") == "y":
                    self.saveModel(model)
                plt.plot(training_losses, label='Training Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()



    
if __name__ == '__main__':
    script = Main()
    script.startNN()
