import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Creating the dataset with Lysine, Moco, Sam and Purine


def Create_Data(Path, Data, Output, Type):
    with open(Path) as csvfile:
        Data_Path = list(csv.DictReader(csvfile))
        for x in Data_Path:
            # Creating the feature vector of mono and di nucleotides
            Data.append([x["A"], x["T"], x["G"], x["C"], x["AA"], x["AC"], x["AG"], x["AU"], x["CA"], x["CC"],
                         x["CG"], x["CU"], x["GA"], x["GC"], x["GG"], x["GU"], x["UA"], x["UC"], x["UG"], x["UU"]])
            Output.append(Type)
        return Data, Output

# Converting the values to Float for Mathematical purposes


def Convert_to_Float(Data, Output):
    for i in range(len(Data)):
        for j in range(20):
            Data[i][j] = float(Data[i][j])
        Output[i] = float(Output[i])
    return Data, Output


Data = []
Output = []

# Exporting the CSV paths
lysinePath = 'Dataset/lysine_frequency.csv'
mocoPath = 'Dataset/moco_frequency.csv'
purinePath = 'Dataset/purine_frequency.csv'

samPath = 'Dataset/sam_frequency.csv'
GlmSPath = 'Dataset/GlmS.csv'
cobalaminPath = 'Dataset/cobalamin.csv'

Cyclicgmp1Path = 'Dataset/Cyclicgmp1.csv'
FMNPath = 'Dataset/FMN.csv'

preq1Path = 'Dataset/preq1.csv'
glycinePath = 'Dataset/Glycine.csv'

preq12Path = 'Dataset/preq12.csv'
tppPath = 'Dataset/tpp.csv'

ykokPath = 'Dataset/ykok.csv'
saHPath = 'Dataset/saH.csv'

SAMPath = 'Dataset/SAM.csv'
sam2Path = 'Dataset/sam2.csv'


Data, Output = Create_Data(lysinePath, Data, Output, 1)
Data, Output = Create_Data(mocoPath, Data, Output, 2)
Data, Output = Create_Data(purinePath, Data, Output, 3)

Data, Output = Create_Data(samPath, Data, Output, 4)

Data, Output = Create_Data(GlmSPath, Data, Output, 5)
Data, Output = Create_Data(cobalaminPath, Data, Output, 6)

Data, Output = Create_Data(Cyclicgmp1Path, Data, Output, 7)
Data, Output = Create_Data(FMNPath, Data, Output, 8)

Data, Output = Create_Data(preq1Path, Data, Output, 9)
Data, Output = Create_Data(glycinePath, Data, Output, 10)

Data, Output = Create_Data(preq12Path, Data, Output, 11)
Data, Output = Create_Data(tppPath, Data, Output, 12)

Data, Output = Create_Data(ykokPath, Data, Output, 13)
Data, Output = Create_Data(saHPath, Data, Output, 14)

Data, Output = Create_Data(SAMPath, Data, Output, 15)
Data, Output = Create_Data(sam2Path, Data, Output, 16)


Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output)

# Converting the train data into Float
Data_train, Output_train = Convert_to_Float(Data_train, Output_train)

# Converting the test data into Float
Data_test, Output_test = Convert_to_Float(Data_test, Output_test)

# Preprocessing the data
scaler = StandardScaler()
scaler.fit(Data_train)

Data_train = scaler.transform(Data_train)
Data_test = scaler.transform(Data_test)

# Classification using tanh activation function
mlp = MLPClassifier(solver='sgd', alpha=0.05, learning_rate='adaptive',
                    hidden_layer_sizes=(10, 10), max_iter=1000, momentum=0.99, activation='relu', verbose=False)

# Training the model using Training Data
mlp.fit(Data_train, Output_train)

# Predicting the Output using the Test Data
predictions = mlp.predict(Data_test)

# Printing the Predictions and the Output Labels
#print (predictions, Output_test)

# Confusion Matrix
print(confusion_matrix(Output_test, predictions))

# Classification Report
print(classification_report(Output_test, predictions))
