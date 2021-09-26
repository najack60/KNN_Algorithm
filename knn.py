#-------------------------------------------------------------------------
# AUTHOR: Nate Colbert
# FILENAME: knn.py
# SPECIFICATION: implements the knn algorithm
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3-5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

classPredict = []
#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X =[]
    for j in range(len(db)):
        for k in range(2):
            db[j][k] = float(db[j][k])

    for j in range(len(db)):
        if db[j] == db[i]:
            continue
        X.append(db[j][:2])
    #print(X, "\n\n")

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    yTemp = []

    transform = {'+': 1, '-': 2}

    for j in range(len(db)):
        yTemp.append(transform[db[j][2]])
    #print(yTemp, "\n\n")

    for j in range(len(db)):
        yTemp[j] = float(yTemp[j])
    #print(yTemp, "\n\n")

    for j in range(len(db)):
        if db[j] == db[i]:
            continue
        Y.append(yTemp[j])
    #print(Y, "\n\n")

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = db[i]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([[db[i][0], db[i][1]]])[0]
    classPredict.append(class_predicted)

#print(classPredict)
    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    wrong = 0
    accuracy = 0
    dbNums = []
    for j in range(len(db)):
        dbNums.append(transform[db[j][2]])

    for j in range(len(classPredict)):
        if classPredict[j] != dbNums[j]:
            wrong += 1

    accuracy = (wrong / 10) * 100
#print the error rate
#--> add your Python code here
print("LOO-CV error rate for 1NN: ", accuracy, "%")





