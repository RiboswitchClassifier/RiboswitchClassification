# RiboswitchClassification

[riboflow](https://test.pypi.org/project/riboflow/) is a python package for classifying putative riboswitch sequences into one of 24 classes with > 99% accuracy. It is based on a [tensorflow](https://www.tensorflow.org) deep learning model. ``riboflow`` has been tested using ``Python 3.5.2``. 

ABOUT
------------

RocAucAllBaselineModels.py

    > python3  RocAucAllBaselineModels.py
    
    1. Create's a Picked Model for each of the sklearn classifers stated below:
        AdaBoostClassifier(),
        GaussianNB(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier()
    2. Each model is used on the test set to obtain accuracy, generate a classication report and the ROC-AUC values for each 
       of the 24 classes. 
    3. The MLPClassifier() proved to be the best among the chosen sklearn classifiers and hence Neural Networks (CNN and RNN) 
       were explored further to acheive greater accuracy.
 
    
