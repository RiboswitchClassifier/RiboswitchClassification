[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3334316.svg)](https://doi.org/10.5281/zenodo.3334316)


<script type="text/javascript" src="https://d1bxh8uas1mnw7.cloudfront.net/assets/embed.js">
</script>
<div class="altmetric-embed" data-badge-type="donut" data-altmetric-id="85787699">
</div>


# RiboswitchClassification

[riboflow](https://test.pypi.org/project/riboflow/) is a python package for classifying putative riboswitch sequences into one of 32 classes with > 99% accuracy. It is based on a [tensorflow](https://www.tensorflow.org) deep learning model. ``riboflow`` has been tested using ``Python 3.5.2``. 
The pip package was derived from this source code. This source code of ``rnnApp.py`` and ``cnnApp.py`` can easily be altered to help achieve better accuracy when the riboswitch labels change or the number of classes increase / decrease when constructing the dataset, by changing the number of NN layers, hyperparameters.  

Datasets, Models, Utility Files
------------

1.original_datasets

    a. 32_riboswitches_fasta    --> Fasta   Format of the 32 riboswitches
    b. 32_riboswitches_new_csv  --> CSV     Format of the 32 riboswitches
    
2.processed_datasets

    a. final_32classes.csv                --> Original 32 riboswitches Dataset cleaned and Frequencies calculated
    b. final_32train.csv          --> 90% of each riboswitch class's instances in the final_32classes.csv
    c. final_32test.csv           --> Remaining 10% of each riboswitch class's instances in the final_32classes.csv
    
3.models

    Contains the rnn and cnn model's in h5 format 
    
4.preprocess.py

    Contains various utilities for train:test splitting of the dataset, loading the datasets and other preprocessing of the data. Could be used to generate -mer frequencies, final_train.csv, final_test.csv and used for data preprocessing by all the models (i.e, both base and deep learning models: baseModels.py, rnnApp.py and cnnApp.py) 
    
5.multiclassROC.py

    Used for the ROC analysis of all models (i.e, base and the deep learning models). 
    
6.dynamic.py
    
    Implements a routine to enable dynamic deep learning for new riboswitch classes. Could be used on riboswitch fasta files of any number of classes to generate the equivalent processed csv files having the sequence and k-mer (for now, mono and di-) frequencies (this file can be used by baseModels.py, rnnApp.py, cnnApp.py for training purposes) 
    
sklearn Base Models 
------------

    > python3  baseModels.py
    
    1. Create's a Picked Model for each of the sklearn classifers stated below:
        AdaBoostClassifier(),
        GaussianNB(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier()
    2. Each model is used on the test set to obtain accuracy, generate a classication report and the ROC-AUC values for each 
       of the 32 classes. 
    3. The MLPClassifier() proved to be the best among the chosen sklearn classifiers and hence Neural Networks (CNN and RNN) 
       were explored further to acheive greater accuracy.
 
 keras.tf RNN
------------

    > python3  rnnApp.py
    
    1. Creates a .h5 RNN Model using tensorflow on keras
    2. The model is used on the test set to obtain accuracy, generate a classication report and the ROC-AUC values for each 
       of the 32 classes. 
    3. Provides an Accuracy of 99% on the test set.
    4. New layers and hyperparameter values can be added or changed when dealing with a dataset having different number of classes
    5. The train time is fairly long ( in the magnitude of hours - suitable for system with high specs )
    
keras.tf CNN
------------

    > python3  cnnApp.py
    
    1. Creates a .h5 CNN Model using tensorflow on keras
    2. The model is used on the test set to obtain accuracy, generate a classication report and the ROC-AUC values for each 
       of the 32 classes. 
    3. Provides an Accuracy of 97% on the test set.
    4. New layers and hyperparameter values can be added or changed when dealing with a dataset having different number of classes.
    5. The train time is fairly short ( < 1 min - suitable for low spec systems )
 
    
Authors
----------

Premkumar KeshavAditya R, Bharanikumar Ramit, Palaniappan Ashok. (2019) Classifying riboswitches with >99% accuracy. Microorganisms (to be submitted)

  * Keshav Aditya R.P
    - [Github](https://github.com/KeshavAdityaRP)
    - [LinkedIn](https://www.linkedin.com/in/keshavadityarp/)
  * Ramit Bharanikumar
    - [Github](https://github.com/ramit29)
    - [LinkedIn](https://www.linkedin.com/in/ramit-bharanikumar-12a014114/)
  * Ashok Palaniappan
    - [Senior Assistant Professor](http://www.sastra.edu/staffprofiles/schools/scbt.php?staff_id=C2164)
    - [Github](https://github.com/apalania)
    - [LinkedIn](https://www.linkedin.com/in/ashokpalaniappan/)

Please cite if you use our software.

Copyright & License
-------------------

Copyright (c) 2019, the Authors. MIT License.

