import csv

# Sample Commands
# shuf  testDataset.txt -o testOutputDataset.txt
# shuf  Ribo2WayDatasetP.txt -o Ribo2WayDatasetS.txt
# head -n 270 Ribo2WayDatasetS.txt > Ribo2WayDatasetS.train
# tail -n 23 Ribo2WayDatasetS.txt > Ribo2WayDatasetS.valid

def Create_Data(Path, Data, Output):
    with open(Path) as csvfile:
        Data_Path = list(csv.DictReader(csvfile))
        for x in Data_Path:
                # Creating the feature vector of mono and di nucleotides
                Data.append([x["A"], x["T"], x["G"], x["C"],x["AA"], x["AC"], x["AG"], x["AU"],x["CA"], x["CC"], x["CG"], x["CU"],x["GA"], x["GC"], x["GG"], x["GU"],x["UA"], x["UC"], x["UG"], x["UU"]])
                Output.append(x["Type"])
        return Data, Output

def Convert_to_Float(Data, Output):
    for i in range(len(Data)):
        for j in range(20):
            Data[i][j]=int(Data[i][j])
        Output[i]=int(Output[i])
    return Data, Output

def Construct_Sequence(phase,occurances):  
    term = ""
    flag = 0
    for k in range(occurances):
        term += phase
        flag = 1
    if (flag == 1):    
        term += " "     
    return  term        

def Convert_to_TextFile(Data, Output):
    file = open("FastTextDatasets/Ribo16WayDatasetP.txt","w")
    for i in range(len(Data)):
        sequence = ""
        label = ""
        for j in range(20):
            if (j == 0):
                sequence += Construct_Sequence("a",Data[i][j])
            elif (j == 1):
                sequence += Construct_Sequence("t",Data[i][j]) 
            elif (j == 2):
                sequence += Construct_Sequence("g",Data[i][j]) 
            elif (j == 3):
                sequence += Construct_Sequence("c",Data[i][j]) 
            elif (j == 4):
                sequence += Construct_Sequence("aa",Data[i][j]) 
            elif (j == 5):
                sequence += Construct_Sequence("ac",Data[i][j]) 
            elif (j == 6):
                sequence += Construct_Sequence("ag",Data[i][j]) 
            elif (j == 7):
                sequence += Construct_Sequence("au",Data[i][j])  
            elif (j == 8):
                sequence += Construct_Sequence("ca",Data[i][j]) 
            elif (j == 9):
                sequence += Construct_Sequence("cc",Data[i][j]) 
            elif (j == 10):
                sequence += Construct_Sequence("cg",Data[i][j]) 
            elif (j == 11):
                sequence += Construct_Sequence("cu",Data[i][j]) 
            elif (j == 12):
                sequence += Construct_Sequence("ga",Data[i][j]) 
            elif (j == 13):
                sequence += Construct_Sequence("gc",Data[i][j]) 
            elif (j == 14):
                sequence += Construct_Sequence("gg",Data[i][j]) 
            elif (j == 15):
                sequence += Construct_Sequence("gu",Data[i][j]) 
            elif (j == 16):
                sequence += Construct_Sequence("ua",Data[i][j]) 
            elif (j == 17):
                sequence += Construct_Sequence("uc",Data[i][j])  
            elif (j == 18):
                sequence += Construct_Sequence("ug",Data[i][j]) 
            elif (j == 19):
                sequence += Construct_Sequence("uu",Data[i][j]) 
            else:
                print("Error in Input")    
        # if (Output[i] == 2):
        #     label = "__label__Moco "
        # elif(Output[i] == 3):
        #     label = "__label__Purine "
        # elif(Output[i] == 1):
        #     label = "__label__Lysine "
        # elif(Output[i] == 4):
        #     label = "__label__Sam "
        # else:
        #     print("Error in Output")
        if (Output[i] == 0):
            label = "__label__RF00050 "
        elif(Output[i] == 1):
            label = "__label__RF00059 "
        elif(Output[i] == 2):
            label = "__label__RF00162 "
        elif(Output[i] == 3):
            label = "__label__RF00174 "
        elif(Output[i] == 4):
            label = "__label__RF00234 "
        elif(Output[i] == 5):
            label = "__label__RF00380 "
        elif(Output[i] == 6):
            label = "__label__RF00504 " 
        elif(Output[i] == 7):
            label = "__label__RF00521 "
        elif(Output[i] == 8):
            label = "__label__RF00522 "
        elif(Output[i] == 9):
            label = "__label__RF01051 "
        elif(Output[i] == 10):
            label = "__label__RF01054 "
        elif(Output[i] == 11):
            label = "__label__RF01057 "
        elif(Output[i] == 12):
            label = "__label__Lysine "  
        elif(Output[i] == 13):
            label = "__label__Moco " 
        elif(Output[i] == 14):
            label = "__label__Purine "  
        elif(Output[i] == 15):
            label = "__label__Sam "                                                
        else:
            print("Error in Output")                                 
        file.write(label + sequence + "\n")  
    file.close()    

Data = []
Output = []
# Path1 = 'Frequency_Dataset/lysine_frequency.csv'
# Path4 = 'Frequency_Dataset/sam_frequency.csv'
# Path2 = 'Frequency_Dataset/moco_frequency.csv'
# Path3 = 'Frequency_Dataset/purine_frequency.csv'

Path = 'newDatasets/ribo_dataset_16_classes.csv'

# Data, Output = Create_Data(Path1, Data, Output)
# Data, Output = Create_Data(Path4, Data, Output)
# Data, Output = Create_Data(Path2, Data, Output)
# Data, Output = Create_Data(Path3, Data, Output)

Data, Output = Create_Data(Path, Data, Output)  

Data, Output = Convert_to_Float(Data, Output) 

Convert_to_TextFile(Data, Output)    