from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

count=0 #global variable to always plot different test cases
plots=True #(if set to false, don't plot)

def convertedData(train_set,eq): #eq==1 for 1000 samples for each class, eq==0 for 50 sample for 9 classes and 1000 for 1

    all_indices=np.arange(len(train_set)) #take index of each image
    np.random.shuffle(all_indices)  #shuffle for better variety
    indices=[]
    labels=np.array(train_set['label'][all_indices])

    if eq == 1: #for 1000 samples for each class

        for i in range (len(np.unique(labels))):

            digit_indices=np.where(labels==i)[0] #i[0] since np.where returns tuple
            select_indices=digit_indices[:1000]
            indices.extend(all_indices[select_indices]) #extend to avoid 2D list (append would create 2D list)

    elif eq==0: #50 sample for 9 classes and 1000 for 1 class (5)

        for i in range(len(np.unique(labels))):
            digit_indices = np.where(labels == i)[0]  # i[0] since np.where returns tuple
            select_indices = digit_indices[:50]
            if i==5:
                select_indices = digit_indices[:1000]
            indices.extend(all_indices[select_indices])  # extend to avoid 2D list (append would create 2D list)

    modifiedTrain_set=train_set.select(indices)
    return modifiedTrain_set

def ovoSlow(train_set,test_set,learning_rate,epochs):

    predictions=[]
    classifiers={}
    uniqueLabels=np.unique(train_set['label'])
    lossHistory = []  # list that holds loss for each epoch

    #float instead of unsigned int (default) for negative number support, required in w and b calculations

    #/255.0 to normalize the data and avoid huge leaps in w calculations

    modifiedTrain_set = np.array(train_set['image']).reshape(len(train_set), -1).astype(np.float32) / 255.0
    modifiedTest_set = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32) / 255.0
    labels = np.array(train_set['label'])

    #training

    for i in uniqueLabels:
        for j in uniqueLabels:
            if i > j: #so if 1 gets compared with 2, 2 does not get compared with 1
                # wx + b, initialize w = [0,0...0] and b = 0
                w = np.zeros(modifiedTrain_set.shape[1])
                b = 0

                indices = np.arange(len(modifiedTrain_set)) #take index of each image
                for l in range(epochs):
                     total_epoch_errors=0
                     np.random.shuffle(indices)  # shuffle so different labels are tested each time (otherwise 1000 times 0, 1000 times 1...)
                     for k in indices:
                         if i == labels[k] or j == labels[k]: #take only data of the two classes
                                  #now we need to make the classes separable (one label = 1, other label = -1)
                                 if labels[k] == i:
                                     classLabel=1
                                 else:
                                     classLabel=-1
                                 prediction = w.T @ modifiedTrain_set[k] + b
                                 if classLabel*prediction<=0: #if prediction is in another class than the actual one
                                    total_epoch_errors += 1
                                    w=w+classLabel*modifiedTrain_set[k]*learning_rate
                                    b=b+classLabel*learning_rate

                lossHistory.append(total_epoch_errors)

                classifiers[(i,j)]=(w,b)

    #predictions
    for k in range(len(modifiedTest_set)):
        votes=np.zeros(len(uniqueLabels)) #for each label 0,1...9
        for (i,j), (w,b) in classifiers.items():
            if w.T @ modifiedTest_set[k] + b >=0:
                votes[i]+=1 #vote for i class
            else:
                votes[j]+=1 #vote for j class
        maxIndex=np.argmax(votes)
        predictions.append(maxIndex)
    return predictions, lossHistory

def ovoOptimized(train_set,test_set,learning_rate,epochs):

    predictions = []
    classifiers = {}
    uniqueLabels = np.unique(train_set['label'])
    lossHistory = []  # list that holds loss for each epoch

    # float instead of unsigned int (default) for negative number support, required in w and b calculations

    # /255.0 to normalize the data and avoid huge leaps in w calculations

    modifiedTrain_set = np.array(train_set['image']).reshape(len(train_set), -1).astype(np.float32) / 255.0
    modifiedTest_set = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32) / 255.0
    labels = np.array(train_set['label'])

    # training

    for i_index in range(len(uniqueLabels)):

        for j_index in range(i_index):

            i=uniqueLabels[i_index]
            j=uniqueLabels[j_index]

            mask = (labels==i) | (labels==j) #take only data of the two classes, concurrently

            #mask is true if label i or j is included, else false

            # take only true elements (data of the two classes)
            filteredTrain_set=modifiedTrain_set[mask]

            filteredLabels=labels[mask]

            classLabel=np.where(filteredLabels==i,1,-1) #if equal to i, 1. Else -1

            # wx + b, initialize w = [0,0...0] and b = 0
            w = np.zeros(modifiedTrain_set.shape[1])
            b = 0

            indices=np.arange(len(filteredTrain_set)) #take index of each image
            for l in range(epochs):
                total_epoch_errors = 0
                np.random.shuffle(indices) #shuffle so different labels are tested each time (otherwise 1000 times 0, 1000 times 1...)

                for k in indices:

                        prediction = np.dot(w,filteredTrain_set[k]) + b #order for inner product doesn't matter since both parameters are vectors

                        if classLabel[k] * prediction <= 0:  # if prediction is in another class than the actual one
                            total_epoch_errors += 1
                            w = w + classLabel[k] * filteredTrain_set[k] * learning_rate
                            b = b + classLabel[k] * learning_rate

            lossHistory.append(total_epoch_errors)

            classifiers[(i, j)] = (w, b)

    # predictions

    votes = np.zeros((len(modifiedTest_set),len(uniqueLabels)))  # 2D vector, row is number of images, column is for each label 0,1...9
    for (i, j), (w, b) in classifiers.items():
        prediction_scores = np.dot(modifiedTest_set,w) + b #w needs to be the second parameter for correct dimensions for inner product (10000 x 784) @ (784 x 1)

        votes[prediction_scores>=0,i]+=1 #if prediction[i] >=0, add 1
        votes[prediction_scores<0,j]+=1  #if prediction[j] >=0, add 1

    predictions=np.argmax(votes,axis=1) #axis = 1 to check votes horizontally (which label got the most votes for each image)
    return predictions.tolist(), lossHistory

def ova(train_set,test_set,learning_rate,epochs):

    predictions = []
    classifiers = {}
    uniqueLabels = np.unique(train_set['label'])
    lossHistory = []  # list that holds loss for each epoch

    # float instead of unsigned int (default) for negative number support, required in w and b calculations

    # /255.0 to normalize the data and avoid huge leaps in w calculations

    modifiedTrain_set = np.array(train_set['image']).reshape(len(train_set), -1).astype(np.float32) / 255.0
    modifiedTest_set = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32) / 255.0
    labels = np.array(train_set['label'])

    # training
    for i in uniqueLabels:

        # wx + b, initialize w = [0,0...0] and b = 0
        w = np.zeros(modifiedTrain_set.shape[1])
        b = 0
        classLabel = np.where(labels == i, 1, -1)  # if equal to i, 1. Else -1

        indices = np.arange(len(modifiedTrain_set))  # take index of each image

        for l in range(epochs):
            total_epoch_errors = 0
            np.random.shuffle(indices)  # shuffle so different labels are tested each time (otherwise 1000 times 0, 1000 times 1...)

            for k in indices:

                prediction = np.dot(w, modifiedTrain_set[k]) + b  # order for inner product doesn't matter since both parameters are vectors

                if classLabel[k] * prediction <= 0:  # if prediction is in another class than the actual one
                    total_epoch_errors += 1
                    w = w + classLabel[k] * modifiedTrain_set[k] * learning_rate
                    b = b + classLabel[k] * learning_rate

        lossHistory.append(total_epoch_errors)

        classifiers[i] = (w, b)

    # predictions
    all_scores = np.zeros((len(modifiedTest_set),len(uniqueLabels)))  # 2D vector, row is number of images, column is for each label 0,1...9

    for (i) , (w, b) in classifiers.items():
        prediction_scores = np.dot(modifiedTest_set,w) + b  # w needs to be the second parameter for correct dimensions for inner product (10000 x 784) @ (784 x 1)
        all_scores[:,i]=prediction_scores #add score for each image

    predictions = np.argmax(all_scores,axis=1)  # axis = 1 to check votes horizontally (which label got the most votes for each image)
    return predictions.tolist(), lossHistory

def softmax(z): #helper function that calculates softmax, required in linearClassifier
    #for large z, e^z will me NaN (overflow)
    z_max=np.max(z,axis=1,keepdims=True) #for each row, find max
    numerator = np.exp(z-z_max) #substract max for each z (trick to avoid overflow)
    denominator=np.sum(numerator, axis=1, keepdims=True ) #for each row, calculate sum (keepdims=True, so (10000,1) instead of (10000,)
    return numerator/denominator

def linearClassifier(train_set,test_set,learning_rate,epochs):

    predictions = []
    classifiers = {}
    uniqueLabels = np.unique(train_set['label'])

    lossHistory = [] #list that holds loss for each epoch

    # float instead of unsigned int (default) for negative number support, required in w and b calculations

    # /255.0 to normalize the data and avoid huge leaps in w calculations

    modifiedTrain_set = np.array(train_set['image']).reshape(len(train_set), -1).astype(np.float32) / 255.0
    modifiedTest_set = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32) / 255.0
    labels = np.array(train_set['label'])

    # training, w x + b
    variance = 2 / (784 + 10)  # 1/784, 784 pixels (input), 1/10, 10 unique labels (output) (Xavier method)
    std = np.sqrt(variance)  # standard deviation
    w = np.random.randn(784, 10) * std  # initialize w
    uniqueLabels = np.unique(train_set['label'])  # 0,1,2..,9
    b = np.zeros(len(uniqueLabels))

    #one hot encoding is needed for correct softmax use
    #matrix of one hot encoding is trainingData x unique labels (1000 x 10). For each image we have the probability for it to belong in each class (e.g. 0.1 for 1, 0.2 for 2..., 0.3 for 9)

    actual=np.zeros((len(labels),len(uniqueLabels))) #1000 x 10
    actual[np.arange(len(labels)),labels]=1 #for each row, actual[label]=1 (e.g. label = 5, actual[5] = 1)

    for i in range (epochs):

        z=np.dot(modifiedTrain_set,w) + b # w needs to be the second parameter for correct dimensions for inner product (10000 x 784) @ (784 x 1)
        prediction=softmax(z)

        loss=-np.sum(np.multiply(actual,np.log(prediction +1e-8)))/len(labels) #calculate cross entropy loss (to plot learning curves)
        # add 1e-8 so log(0) isn't undefined
        # /len(labels) for independence in number of samples
        lossHistory.append(loss)

        #calculate loss dZ,dW,dB
        dZ=prediction-actual
        # Z = X * W + b,
        # dz / dw = X, dw = X * dz
        # x is samples x 784 (28 x 28), dZ is samples x 10, dW needs to be 784 x 10, so we take X.T
        dW=np.dot(modifiedTrain_set.T,dZ)/len(labels) #/labels because we need the average

        # dz / db = 1, db = dz
        dB=np.sum(dZ,axis=0)/len(labels) #/labels because we need the average, axis=0 to take average bias for each label

        w=w-learning_rate*dW #go opposite of derivative, to minimize loss (dz)
        b=b-learning_rate*dB #go opposite of derivative, to minimize loss (dz)

    # predictions
    predictions=np.dot(modifiedTest_set,w) + b
    predictions_softmax=softmax(predictions)
    finalPredictions=np.argmax(predictions_softmax,axis=1) #for each row, take the largest probability
    return finalPredictions.tolist(), lossHistory

def testAlgorithm(algorithm,train_set,test_set,learning_rate,epochs): #helper function to test algorithms
    global plots,count
    start = time.time()
    results = algorithm(train_set,test_set,learning_rate,epochs)
    predictions, lossHistory = results[0],results[1]
    end = time.time()
    correct = 0
    true_labels = []  # used for confusion matrix
    for i in range(len(predictions)):
        true_labels.append(test_set[i]['label'])
        if test_set['label'][i] == predictions[i]:
            correct += 1
        if i<1 and plots:
            plt.imshow(test_set[count]['image'], cmap='gray')
            plt.title(f"Predicted: {predictions[count]}, Actual: {test_set[count]['label']}")
            plt.show()
            count+=1
    accuracy = correct / len(predictions) * 100
    print(algorithm.__name__,"accuracy:", round(accuracy,4), "% with learning rate:", learning_rate,", Time taken:", round(end - start,2),"seconds, with epochs:", epochs)
    if plots:
        plt.figure(figsize=(8, 5))
        plt.plot(lossHistory, marker='o', linestyle='-', color='r')
        if algorithm.__name__ == "linearClassifier":
            plt.title(f"Learning Curve: {algorithm.__name__}, with learning rate: {learning_rate}, with epochs: {epochs}")
            plt.xlabel("Epochs")
        elif algorithm.__name__ == "ova":
            plt.title(f"Training errors per classifier: {algorithm.__name__}, with learning rate: {learning_rate}, with epochs: {epochs}")
            plt.xlabel("Binary Classifiers (10 Pairs)")
        elif algorithm.__name__ == "ovoOptimized" or algorithm.__name__ == "ovoSlow":
            plt.title(f"Training errors per classifier: {algorithm.__name__}, with learning rate: {learning_rate}, with epochs: {epochs}")
            plt.xlabel("Binary Classifiers (45 Pairs)")
        plt.ylabel("Loss / Misclassifications")
        plt.grid(True)
        plt.show()
        cm = confusion_matrix(true_labels, predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title(f"Confusion Matrix: {algorithm.__name__}, learning rate: {learning_rate}, epochs: {epochs}")
        plt.show()
    return accuracy



#loading the dataset
dataset=load_dataset('ylecun/mnist')
dataset.set_format(type='numpy')

train_set=dataset['train']
test_set=dataset['test']

#testing the algorithms
convertedTrain_set=convertedData(train_set,1)
print("1000 samples for each class in trainset")
print("Testing algorithm linearClassifier, using different learning rate and epochs")

results=[]
for j in range (50,300,50):
    for i in range (5,20,5):
        accuracy = testAlgorithm(linearClassifier,convertedTrain_set,test_set,i/10.0,j)
        results.append((accuracy,i/10.0,j))
best_acc, best_lr1, best_epochs1 = max(results)
print("Max accuracy: ",round(best_acc,4)," with learning rate: ",best_lr1," with epochs: ",best_epochs1)

accuracies = [item[0] for item in results]
learning_rates = [item[1] for item in results]
epochs_list = [item[2] for item in results]
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter_plot = ax.scatter(learning_rates, epochs_list, accuracies, c=accuracies, cmap='viridis', s=100, alpha=0.8)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Epochs')
ax.set_zlabel('Accuracy (%)')
plt.title('Comparing: Learning Rate vs Epochs vs Accuracy for linearClassifier')
color_bar = fig.colorbar(scatter_plot, ax=ax, pad=0.1)
color_bar.set_label('Accuracy (%)')
plt.show()

print("Testing algorithm One versus one, using different learning rate and epochs")

results=[]
for j in range (50,300,50):
    for i in range (5,20,5):
        accuracy = testAlgorithm(ovoOptimized,convertedTrain_set,test_set,i/10.0,j)
        results.append((accuracy,i/10.0,j))
best_acc, best_lr2, best_epochs2 = max(results)
print("Max accuracy: ",round(best_acc,4)," with learning rate: ",best_lr2," with epochs: ",best_epochs2)

accuracies = [item[0] for item in results]
learning_rates = [item[1] for item in results]
epochs_list = [item[2] for item in results]
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter_plot = ax.scatter(learning_rates, epochs_list, accuracies, c=accuracies, cmap='viridis', s=100, alpha=0.8)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Epochs')
ax.set_zlabel('Accuracy (%)')
plt.title('Comparing: Learning Rate vs Epochs vs Accuracy for ovoOptimized')
color_bar = fig.colorbar(scatter_plot, ax=ax, pad=0.1)
color_bar.set_label('Accuracy (%)')
plt.show()

print("Testing algorithm One versus all, using different learning rate and epochs")

results=[]
for j in range (50,300,50):
    for i in range (5,20,5):
        accuracy = testAlgorithm(ova,convertedTrain_set,test_set,i/10.0,j)
        results.append((accuracy,i/10.0,j))
best_acc, best_lr3, best_epochs3 = max(results)
print("Max accuracy: ",round(best_acc,4)," with learning rate: ",best_lr3," with epochs: ",best_epochs3)

accuracies = [item[0] for item in results]
learning_rates = [item[1] for item in results]
epochs_list = [item[2] for item in results]
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter_plot = ax.scatter(learning_rates, epochs_list, accuracies, c=accuracies, cmap='viridis', s=100, alpha=0.8)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Epochs')
ax.set_zlabel('Accuracy (%)')
plt.title('Comparing: Learning Rate vs Epochs vs Accuracy for ova')
color_bar = fig.colorbar(scatter_plot, ax=ax, pad=0.1)
color_bar.set_label('Accuracy (%)')
plt.show()

print("50 samples for each class in trainset and 1000 samples in class 5")
print("Testing each algorithm, with optimal learning rate and epochs, found in previous tests")
convertedTrain_set=convertedData(train_set,0)
testAlgorithm(linearClassifier,convertedTrain_set,test_set,best_lr1,best_epochs1)
testAlgorithm(ovoOptimized,convertedTrain_set,test_set,best_lr2,best_epochs2)
testAlgorithm(ova,convertedTrain_set,test_set,best_lr3,best_epochs3)
print("Testing slow version of one versus one (accuracy is expected to be the same)")
testAlgorithm(ovoSlow,convertedTrain_set,test_set,best_lr2,best_epochs2)