import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier




def testModle(data_train, data_test, targets_train, targets_test, classifier):
    modle = classifier.fit(data_train, targets_train)
    return modle.predict(data_test)

def displayTest(targets_predicted,targets_test):
    simCount = 0
    difCount = 0
    for i in range(len(targets_predicted)):
        if targets_predicted[i] == targets_test[i]:
            simCount += 1
        if targets_predicted[i] != targets_test[i]:
            difCount += 1
    print("\tsim =\t", simCount)
    print("\tdif =\t", difCount)
    print("\ttotal =\t", len(targets_predicted))
    print("\t\t%",simCount/len(targets_predicted) * 100)
    return simCount/len(targets_predicted) * 100


class KNNClasifier:
    def __init__(self):
        pass
    
    def fit(self, data_train, targets_train):
        return KNNModle(data_train, targets_train)
    
def eDistance(a, b):
    out = 0
    for point in range(len(a)):
        out = out + (a[point]-b[point])**2
    return out
    
    
class KNNModle:
    def __init__(self,data_train, targets_train):
        self.data_train    = data_train
        self.targets_train = targets_train
        self.k = 3
        return None
    
    def getNebors(self, data_test_row):
        nebors = []
        distances = []
        
        #find how far everything is from eachother
        for data_train_row in self.data_train:
            #distances.append(np.asscalar(LA.norm(data_train_row - data_test_row)))  #Euclidean distance: LA.norm(row - data_train_row
            distances.append(eDistance(data_train_row, data_test_row))
        #combine rlivent data
        nebors = list(zip(distances, self.targets_train, self.data_train))
        
        #sort based on distance
        nebors = sorted(nebors, key=lambda k:k[0])          
        # cut so their are only k number # default is 3
        nebors = nebors[:self.k]     
        return nebors
    
    def determinSimilarity(self, nebors):
        # get list of top targets
        possabilities = list(zip(*nebors))[1]
        #find unique values of top targets
        targetRange = set(possabilities)
        count = []
        #loop through values of top targets and count how many there are 
        for num in targetRange:
            count.append(list(possabilities).count(num))
        # combine values and counts
        out = list(zip(targetRange, count))
        # sort
        out = sorted(out, key=lambda k:k[1], reverse=True)
        # return 
        return out[0][0]
    
    def predict(self, data_test):
        targets = []
        for data_test_row in data_test:
            nebors = self.getNebors(data_test_row)
            targets.append(self.determinSimilarity(nebors))
        return targets





def main(argv):
    testCount = 1000
    
    sklearnKnnSum = 0
    myKNNsum = 0
    for i in range(testCount):
        data_train, data_test, targets_train, targets_test = train_test_split(datasets.load_iris().data, datasets.load_iris().target, test_size = .3)
        print("sklearn KNN")
        classifier = KNeighborsClassifier(n_neighbors=3)
        targets_predicted = testModle(data_train, data_test, targets_train, targets_test, classifier)
        sklearnKnnSum += displayTest(targets_predicted,targets_test)
        
        print("My KNN")
        classifier = KNNClasifier()
        targets_predicted = testModle(data_train, data_test, targets_train, targets_test, classifier)
        myKNNsum += displayTest(targets_predicted,targets_test)
    print ("My KNN Average : ", myKNNsum / testCount)
    print ("sklearn KNN Average : ", sklearnKnnSum / testCount)
    print ("difference :", abs((myKNNsum / testCount) - (sklearnKnnSum / testCount)) )
    return None




if __name__ == "__main__":
    main(sys.argv)