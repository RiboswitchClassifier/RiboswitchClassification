import matplotlib.pyplot as plt
import numpy as np

def metrics(matrix):
    i=0
    TP=[]
    FN=[]
    AP=[]
    FP=[]
    TN=[]
    AN=[]
    Recall=[]                  #Recall is the true positive rate
    FPR=[]
    Precision=[]
    Accuracy=[]
    F1=[]
    sum_matrix=0
    for j in range(len(matrix)):
        for k in range(len(matrix)):
            sum_matrix+=matrix[j][k]
    #print("sum matrix:{}".format(sum_matrix))
    while i < len(matrix):
        tp=matrix[i][i]
        TP.append(tp)
        sum = 0
        for j in range(len(matrix[i])):
            sum+=matrix[i][j]
        fn = sum - tp
        AP.append(sum)
        FN.append(fn)
        sum_col = 0
        for j in range(len(matrix[i])):
            sum_col+=matrix[j][i]
        fp = sum_col - tp
        FP.append(fp)
        tn = sum_matrix - tp - fp - fn
        TN.append(tn)
        an = fp + tn
        AN.append(tn)
        i=i+1

    return TP, FN, AP, FP, TN, AN

def choose_from_confusion_matrix(confu):
    True_Positives={}
    False_Negatives={}
    All_Positives={}
    False_Positives={}
    True_Negatives={}
    All_Negatives={}
    for i,j in confu.items():
        True_Positives[i], False_Negatives[i], All_Positives[i], False_Positives[i], True_Negatives[i], All_Negatives[i] = metrics(j) 
        # True_Positives[i] = metrics(j)[0]
        # False_Negatives[i] = metrics(j)[1]
        # All_Positives[i] = metrics(j)[2]
        # False_Positives[i] = metrics(j)[3]
        # True_Negatives[i] = metrics(j)[4]
        # All_Negatives[i] = metrics(j)[5] 
    print (True_Positives, False_Negatives, All_Positives, False_Positives, True_Negatives, All_Negatives)
    return True_Positives, False_Negatives, All_Positives, False_Positives, True_Negatives, All_Negatives

def Rec(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = float(a)/float(b)
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j
    return Recal

def Pre(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = float(a)/(float(a)+float(b))
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal

def Acc(w,x,y,z,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    c=[]
    d=[]
    for i,j in w.items():
        names.append(i)
        a.append(j)
    for i,j in x.items():
        b.append(j)
    for i,j in y.items():
        c.append(j)
    for i,j in z.items():
        d.append(j)
    for i,j,k,l in zip(a,b,c,d):
        rec=[]
        for a,b,c,d in zip(i,j,k,l):
            recall = (float(a)+float(b))/(float(a)+float(b)+float(c)+float(d))
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recall[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal

def F(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = (2*float(a)*float(b))/(float(a)+float(b))
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal

def fdr(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = float(a)/float(b)
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal

def display_graphs(Precision, Recall, Accuracy, F1, FPR):
    metric=[]
    metric.append(Precision.copy())
    metric.append(Recall.copy())
    metric.append(Accuracy.copy())
    metric.append(F1.copy())
    params=['Precision','Recall','Accuracy','F1']
    print (metric)
    for i,j in zip(metric,params):
        plt.rcdefaults()
        fig, ax = plt.subplots()
        algorithms=list(i.keys())
        y_pos = np.arange(len(algorithms))
        Precison=list(i.values())
        ax.barh(y_pos, Precison, align='center',color='green')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(algorithms,rotation=55)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(j)
        plt.xticks(np.arange(0,1,0.1))
        ax.set_title('{} for the different classifiers'.format(j))

        plt.show()

        #  create the figure
    plt.rcdefaults()
    fig, ax = plt.subplots()
    algorithms=list(FPR.keys())
    y_pos = np.arange(len(algorithms))
    Precison=list(FPR.values())
    ax.barh(y_pos, Precison, align='center',color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(algorithms,rotation=55)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(j)
    plt.xticks(np.arange(0,0.01,0.01))
    ax.set_title('FPR for the different classifiers')

    plt.show()
