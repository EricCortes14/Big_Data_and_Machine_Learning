#import packages
import os
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#################################Assignment 1#################################


#this function displays the images
def displayImages(address):

    #loop through images
    for filename in os.listdir(address):

        #obtain address
        imageAddress = os.path.join(address, filename)

        #obtain image
        image = cv2.imread(imageAddress)

        #print the filename
        print(filename)

        #display the color channels
        plt.imshow(image[:,:,0])
        plt.imshow(image[:,:,1])
        plt.imshow(image[:,:,2])
        plt.show()

        #convert to grayscale.
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #plot grey image
        plt.imshow(imageGray, cmap = plt.get_cmap('gray'))
        plt.show()

        #obtain size of grey image and display
        height, width = imageGray.shape
        print("Image Height:", height)
        print("Image Width:", width, "\n")

        #make new address
        splitAddress = address.split('/')
        fruit = splitAddress[3]
        newLocation = "../data/gray/" + fruit
        newAddress = os.path.join(newLocation, filename)

        #save image in file
        cv2.imwrite(newAddress, imageGray)

#this function resizes the images
def resizeImages(address):

    #loop through images
    for filename in os.listdir(address):

        #obtain address
        imageAddress = os.path.join(address, filename)

        print(filename)

        #obtain image
        image = cv2.imread(imageAddress)

        #plot image
        plt.imshow(image)
        plt.show()

        #get width and height
        height, width = image.shape[:2]

        #obtain scaling ratio
        ratio = 256/float(height)

        #obtain new width
        newWidth = int(width * ratio)

        #obtain number of pixels which have to be removed to be divisable by 8
        widthPixelRemoval = newWidth % 8

        #remove excess pixels from width
        newWidth -= widthPixelRemoval

        #get resized image
        resizedImage = cv2.resize(image, dsize = (newWidth, 256), interpolation = cv2.INTER_CUBIC)

        #display resized image
        plt.clf()
        plt.imshow(resizedImage)
        plt.show()

        #print before and after width and height
        print("Before:\n", "Height:", height, "Width:\n", width)
        print("\nAfter:\n", "Height: 256", "Width:", newWidth)

        #make new address
        splitAddress = address.split('/')
        fruit = splitAddress[3]
        newLocation = "../data/resized/" + fruit
        newAddress = os.path.join(newLocation, filename)

        #save image in file
        cv2.imwrite(newAddress, resizedImage)

#this function generates the vectors of the images
def generateVectors(address, label):

    #loop through images
    for filename in os.listdir(address):

        #obtain image location
        imageLocation = os.path.join(address, filename)

        #obtain image
        tmp = cv2.imread(imageLocation)

        #convert image to 2D
        image = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        #obtain height and width
        height, width = image.shape[:2]

        # Create feature vectors and labels
        rows = round(((height-8)*(width-8))/64)
        flat = label * np.ones((rows, 65), np.uint8)
        k = 0
        for i in range(0,height-8,8):
            for j in range(0,width-8,8):
                crop_tmp1 = image[i:i+8,j:j+8]
                flat[k,0:64] = crop_tmp1.flatten()
                k = k + 1

        #make a dataset of the features
        featureSpace = pd.DataFrame(flat)

        #print the head of the dataset created
        print(featureSpace.head())

        #obtain the image number
        splitFilename= filename.split('.')
        imageNumber = splitFilename[0]

        #obtain the fruit name
        splitAddress = address.split('/')
        fruitName = splitAddress[3]

        #save feature dataset to new file location
        featureSpace.to_csv("../data/vectors/" + fruitName + "/" + imageNumber + "_features.csv", index=False)

#this function displays the images
def displayStatistics(address):

    #loop through images
    for filename in os.listdir(address):

        #obtain image location
        location = os.path.join(address, filename)

        #obtain the image number
        splitFilename= filename.split('.')
        imageNumber = splitFilename[0]

        #obtain the fruit name
        splitAddress = address.split('/')
        fruitName = splitAddress[3]

        #make dataframe label
        dfLabel = fruitName + " " + imageNumber + ":"

        #print fruit and number
        print("\n" + dfLabel)

        #read the dataset with pandas and make a dataframe
        df = pd.read_csv(location)

        #print number of observations of dataframe
        print("Number of observations:", df.shape[0])

        #print dimension of dataframe
        print("\nDimension:", df.ndim)

        #print mean of each feature
        featureMean = df.mean(axis=1)
        print("\nFeature Means: ")
        print(featureMean)

        #print the histogram of the dataframe
        plt.hist(df.to_numpy().flatten())
        plt.show()

#function to create the featue space
def createFeatureSpace(addressList):

    #list to hold all the locations of the files for each address passed in
    location1 = []
    location2 = []
    location3 = []


    #loop through the files of address 1
    for filename in os.listdir(addressList[0]):

        #add location to corresponding list
        location1.append(os.path.join(addressList[0], filename))

    #loop through the files of address 2
    for filename in os.listdir(addressList[1]):

        #add location to corresponding list
        location2.append(os.path.join(addressList[1], filename))

    #loop through the files of address 3
    for filename in os.listdir(addressList[2]):

        #add location to corresponding list
        location3.append(os.path.join(addressList[2], filename))


    ###############################################################################

    #obtain the number of iterations the feature space loop can have
    minLoopIterations = min(len(location1), len(location2))

    #loop through the length of the minLoopIterations
    for x in range(0, minLoopIterations):

        #read the dataset with pandas and make a dataframe of location 1
        df1 = pd.read_csv(location1[x])

        #read the dataset with pandas and make a dataframe of location 2
        df2 = pd.read_csv(location2[x])

        #make a list of the two dataframes
        frames = [df1, df2]

        #concat both dataframes into one using pandas
        mergedFrames = pd.concat(frames)

        #get indexes of the new dataset by using an numpy array of the length of the new dataset
        index = np.arange(len(mergedFrames))

        #randomly mix the indexes
        randomMerged = np.random.permutation(index)

        #finalize random merged dataset
        randomMerged = mergedFrames.sample(frac=1).reset_index(drop=True)

        #print the head of the final dataset
        print(randomMerged.head())

        #write file to computer
        randomMerged.to_csv('../data/feature_space/01/image01_' + str(x + 1) + ".csv", index=False)

    ###############################################################################


    #obtain the number of iterations the feature space loop can have
    minLoopIterations = min(len(location1), len(location2), len(location3))

    #loop through the length of the minLoopIterations
    for x in range(0, minLoopIterations):

        #read the dataset with pandas and make a dataframe of location 1
        df1 = pd.read_csv(location1[x])

        #read the dataset with pandas and make a dataframe of location 2
        df2 = pd.read_csv(location2[x])

        #read the dataset with pandas and make a dataframe of location 2
        df3 = pd.read_csv(location3[x])

        #make a list of the two dataframes
        frames = [df1, df2, df3]

        #concat both dataframes into one using pandas
        mergedFrames = pd.concat(frames)

        #get indexes of the new dataset by using an numpy array of the length of the new dataset
        index = np.arange(len(mergedFrames))

        #randomly mix the indexes
        randomMerged = np.random.permutation(index)

        #finalize random merged dataset
        randomMerged = mergedFrames.sample(frac=1).reset_index(drop=True)

        #print the head of the final dataset
        print(randomMerged.head())

        #write file to computer
        randomMerged.to_csv('../data/feature_space/012/image012_' + str(x + 1) + ".csv", index=False)

#function to display subspaces
def displaySubspaces(address):

    #loop to iterate through all the files in the address
    for filename in os.listdir(address):

        #obtain file location
        location = os.path.join(address, filename)

        #read csv file from location and convert to dataframe
        df = pd.read_csv(location)

        #extract the 3 feature columns and the result column
        x = df['1']
        y = df['54']
        z = df['59']
        r = df['64']

        #create a 2D scatter plot of 2 features and plot
        plt.scatter(x, y, c = r)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 54')
        plt.show()

        #print label and filename
        print("2D Plot of File: " + filename)

        #create a 3D scatter plot of 3 features and plot
        ax = plt.axes(projection = "3d")
        ax.scatter3D(x, y, z, c = r)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 54')
        ax.set_zlabel('Feature 59')
        plt.show()

        #print label and filename
        print("3D Plot of File: " + filename)
        
        
#################################Assignment 2##################################


#function to slit the data into training and  as well as display graphs
def dataSplit(address):


    #loop to iterate through all the files in the address
    for filename in os.listdir(address):
        
        #obtain file location
        location = os.path.join(address, filename)

        #read csv file from location and convert to dataframe
        featureSpace = pd.read_csv(location, header = None)
        
        #convert the feature space into a numpy array
        featureSpace = np.array(featureSpace)
        
        #obtain the number of rows and columns of the feature space
        rows, columns = featureSpace.shape
        
        #obtain the number that divides the data into training and testing 
        divider = round(rows * 0.8)
        
        #obtain the training data
        trainData = featureSpace[0:divider - 1, :]
        
        #obtain testing data
        testData = featureSpace[divider:rows, :]
        
        #conversion of trainnig data to data frame
        trainData = pd.DataFrame(trainData)
        
        #convertion of testing data to data frame
        testData = pd.DataFrame(testData)
        
        #obtain the classifier folder name
        splitAddress= address.split('/')
        classifierNumber = splitAddress[3]
        
        #save the training data
        trainData.to_csv("../data/data_split/train/" + classifierNumber + "/" + filename, index=False, header = None)
        
        #save the testing data
        testData.to_csv("../data/data_split/test/" + classifierNumber + "/" + filename, index=False, header = None)
        
        #obtain feature 32 for both train and test
        trainFeature1 = np.array(trainData[31])
        testFeature1 = np.array(testData[31])
        
        #obtain feature 59 for both train and test
        trainFeature2 = np.array(trainData[58])
        testFeature2 = np.array(testData[58])
        
        #obtain the labels for both train and test
        trainLabel = np.array(trainData[64])
        testLabel = np.array(testData[64])
        
        #show histogram of feature 32
        plt.hist(trainFeature1, label = "Train")
        plt.hist(testFeature1, label = "Test")
        plt.legend(loc = "upper right")
        plt.title("Feature 32")
        plt.show()
        
        #show histogram of feature 59
        plt.hist(trainFeature2, label = "Train")
        plt.hist(testFeature2, label = "Test")
        plt.legend(loc = "upper right")
        plt.title("Feature 59")
        plt.show()
        
        #create a 2D scatter plot of 2 features and plot (train)
        plt.scatter(trainFeature1, trainFeature2, c = trainLabel)
        plt.xlabel('Feature 32')
        plt.ylabel('Feature 59')
        plt.title("Train Data")
        plt.show()
        
        #create a 2D scatter plot of 2 features and plot (train)
        plt.scatter(testFeature1, testFeature2, c = testLabel)
        plt.xlabel('Feature 32')
        plt.ylabel('Feature 59')
        plt.title("Test Data")
        plt.show()

#function to create a lasso regressions
def lassoRegression(address):
    
    #print the header for the function
    print("\nLasso Regression of Two-Class Classifier (per spreadsheet):")

    #loop to iterate through all the files in the address
    for filename in os.listdir(address):
        
        #obtain the file location
        location = os.path.join(address, filename)

        #read the csv file from location and convert it into a dataframe
        featureSpace = pd.read_csv(location, header = None)
        
        #establish lambda value
        L = 0.1
        
        #extract the label column
        labels = featureSpace[64]
        
        #extract the feature vectors 
        vectors = featureSpace.drop(64, axis=1, inplace = False)
        
        #convert the feature vectors into a numpy array
        x = np.array(vectors)
        
        #convert the label column into a numpy array
        y = np.array(labels)
        
        #obtain the number of rows and columns of the feature vectors
        rows, columns = vectors.shape
        
        #obtain the number that divides the data into training and testing 
        divider = round(rows * 0.8)
        
        #obtain the training data
        x_train = x[0:divider - 1, :]
        y_train = y[0:divider - 1]
        
        #obtain testing data
        x_test = x[divider:rows, :]
        y_test = y[divider:rows]
        
        #convert the features into a matrix using a Numpy array
        X1 = np.array(x_train)
        
        #transpose the feature matrix
        X2 = X1.transpose()
        
        #multiply feature matrix with the transposed feature matrix
        XX = np.matmul(X2, X1)
        
        #compute the inverse of the multiplied matricies
        IX = inv(XX)
        
        #multiply inverse matrix with original matrix
        TX = np.matmul(X1, IX)
        
        #convert the label column into a matrix using Numpy array
        Y1 = np.array(y_train)
        
        #transpose the label matrix
        Y2 = Y1.transpose()
        
        #multiply transposed label matrix with TX
        A1 = np.matmul(Y2, TX)
        
        #obtain S for lasso regression function
        S = np.sign(A1)
        
        #multiply S with lambda/2
        SL = S * (L / 2)
        
        #calculate yy*xx'
        A2 = np.matmul(Y2, X1)
        
        #subtract previous matrix by SL
        A3 = A2 - SL
        
        #obtain A matrix
        A = np.matmul(A3, IX)
        
        #multiply the test vectors by the A matrix
        ZZ1 = np.matmul(x_test, A)
        
        #determine which positions are above the mean
        ZZ2 = ZZ1 > ZZ1.mean()
        
        #convert true/false to 1 or 0
        y_pred = ZZ2.astype(int)
        
        #make a confusion matrix out of the testing data
        confusionMatrix = confusion_matrix(y_test, y_pred)
        
        #compute false positive, false negative, and true positive
        FP = confusionMatrix[0,1]
        FN = confusionMatrix[1,0]
        TP = confusionMatrix[1,1]
        
        #print name of file
        print("\n" + filename + ":")
        
        #compute and print the precision quality measure
        precision = 1 / (1 + (FP / TP))
        print("Precision Score:", precision)
        
        #compute and print the sensitivity quality measure
        sensitivity = 1 / (1 + (FN / TP))
        print("Sensitivity Score:", sensitivity)
        
        #multiply the entire set of vectors by the A matrix
        ZZ1 = np.matmul(x, A)
        
        #determine which positions are above the mean
        ZZ2 = ZZ1 > ZZ1.mean()
        
        #convert true/false to 1 or 0
        y_pred2 = ZZ2.astype(int)
        
        #make the first row of the original dataframe as the column names
        featureSpace.columns = featureSpace.iloc[0]
        
        #remove first row of the original dataframe
        featureSpace = featureSpace[1:]
        
        #create a dataframe of the predicted results of all of the vectors
        predDataFrame = pd.DataFrame(y_pred2, columns = ['65'])
        
        #join the predicted dataframe to original dataframe
        newFeatureSpace = featureSpace.join(predDataFrame)
        
        #obtain the image number
        splitFilename= filename.split('_')
        imageNumber = splitFilename[1]

        #save feature dataset to new file location
        newFeatureSpace.to_csv("../data/new_feature_space/01/image01_" + imageNumber, index=False)
        
        #conversion of confusion matrix to a dataframe
        confusionMatrix = pd.DataFrame(confusionMatrix)
        
        #save confusion matrix to new file location
        confusionMatrix.to_csv("../data/confusion_matrix/01/image01_" + imageNumber, index=False)

#function to create random forest classifiers
def randomForest(address):
        
    #print header for function
    print("\nRandom Forest Classifier of Three-Class Classifier (per spreadsheet):")

    #loop to iterate through all the files in the address
    for filename in os.listdir(address):
        
        #obtain file location
        location = os.path.join(address, filename)

        #read csv file from location and convert to dataframe
        featureSpace = pd.read_csv(location, header = None)
        
        #extract the label column
        labels = featureSpace[64]
        
        #extract the feature vectors 
        vectors = featureSpace.drop(64, axis=1, inplace = False)
        
        #convert the feature vectors into a numpy array
        x = np.array(vectors)
        
        #convert the label column into a numpy array
        y = np.array(labels)
        
        #obtain the number of rows and columns of the feature vectors
        rows, columns = vectors.shape
        
        #obtain the number that divides the data into training and testing 
        divider = round(rows * 0.8)
        
        #obtain the training data
        x_train = x[0:divider - 1, :]
        y_train = y[0:divider - 1]
        
        #obtain testing data
        x_test = x[divider:rows, :]
        y_test = y[divider:rows]
        
        #create the random forest classifier model
        classifier = RandomForestClassifier(random_state = 0, n_estimators = 200, n_jobs = -1)
        
        #train the model with the training data
        model = classifier.fit(x_train, y_train)
        
        #predict the test vectors
        y_pred = model.predict(x_test)
        
        #create a confusion matrix
        confusionMatrix = confusion_matrix(y_test, y_pred)
        
        #obtain the true negative, false positive, false negative, and true positive 
        FP = confusionMatrix[1,0]
        FN = confusionMatrix[0,1]
        TP = confusionMatrix[0,0]
        
        #print name of file
        print("\n" + filename + ":")
                
        #compute and print the precision quality measure
        precision = 1 / (1 + (FP / TP))
        print("Precision_Score:", precision)
        
        #compute and print the sensitivity quality measure
        sensitivity = 1 / (1 + (FN / TP))
        print("Sensitivity_Score:", sensitivity)
        
        #use model to predict for all vectors
        y_pred2 = model.predict(x)
        
        #make the first row of the original dataframe as the column names
        featureSpace.columns = featureSpace.iloc[0]
        
        #remove first row of the original dataframe
        featureSpace = featureSpace[1:]
        
        #create a dataframe of the predicted results of all of the vectors
        predDataFrame = pd.DataFrame(y_pred2, columns = ['65'])
        
        #join the predicted dataframe to original dataframe
        newFeatureSpace = featureSpace.join(predDataFrame)
        
        #obtain the image number
        splitFilename= filename.split('_')
        imageNumber = splitFilename[1]
        
        #save new feature dataset to new file location
        newFeatureSpace.to_csv("../data/new_feature_space/012/image012_" + imageNumber, index=False)
        
        #conversion of confusion matrix to a dataframe
        confusionMatrix = pd.DataFrame(confusionMatrix)
        
        #save confusion matrix to new file location
        confusionMatrix.to_csv("../data/confusion_matrix/012/image012_" + imageNumber, index=False)
        
#function to evaluate models
def modelEvaluation(address):
    
    #print the header for the function
    print("\nSklearn Evaluation Metrics (per spreadsheet):")

    #loop to iterate through all the files in the address
    for filename in os.listdir(address):
        
        #obtain the file location
        location = os.path.join(address, filename)

        #read the csv file from location and convert it to a dataframe
        featureSpace = pd.read_csv(location, header = None)
        
        #extract the true label column
        trueLabels = featureSpace[64]
        
        #extract the predicted label column
        predLabels = featureSpace[65]
        
        #convert the true labels into a numpy array
        y_true = np.array(trueLabels)
        
        #convert the predicted labels into a numpy array
        y_pred = np.array(predLabels)
        
        #print the name of the file
        print("\n" + filename + ":")
        
        #print precision evaluation
        print("Precision:", metrics.precision_score(y_true, y_pred, average = 'micro'))

        #print sensitivity evaluation
        print("Sensitivity:", metrics.recall_score(y_true, y_pred, average = 'micro'))

#assignment 1 task 3
displayImages("../data/original/acerolas")
displayImages("../data/original/olives")
displayImages("../data/original/passionfruit")

#assignment 1 task 4
resizeImages("../data/gray/acerolas")
resizeImages("../data/gray/olives")
resizeImages("../data/gray/passionfruit")

#assignment 1 task 5
generateVectors("../data/resized/acerolas", 0)
generateVectors("../data/resized/olives", 1)
generateVectors("../data/resized/passionfruit", 2)

#assignment 1 task 7
displayStatistics("../data/vectors/acerolas")
displayStatistics("../data/vectors/olives")
displayStatistics("../data/vectors/passionfruit")

#assignment 1 task 8
addressList = ["../data/vectors/acerolas", "../data/vectors/olives", "../data/vectors/passionfruit"]
createFeatureSpace(addressList)

#assignment 1 task 9
displaySubspaces("../data/feature_space/01")
displaySubspaces("../data/feature_space/012")

#assignment 2 task 1
dataSplit("../data/feature_space/01")
dataSplit("../data/feature_space/012")

#assignment 2 task 2
lassoRegression("../data/feature_space/01")

#assignment 2 task 3
randomForest("../data/feature_space/012")

#assignment 2 task 4
modelEvaluation("../data/new_feature_space/01")
modelEvaluation("../data/new_feature_space/012")
