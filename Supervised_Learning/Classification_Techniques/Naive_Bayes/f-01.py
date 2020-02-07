import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
def count_types_output(y):              # This function calculates the different types of
    a=[]                                # outputs that are present in the output column.
    for i in y:                         # eg:- It will calculate and result [0,1](malignant/bennie) for cancer
        if i in a:                      # data set and [0,1](survived/not survived) for the titanic data set.
            continue
        else:
            a.append(i)
    return a


def output_counter(y):                              # This function calculates the number of occurrences of the
    output_label = count_types_output(y)            # different types of outputs present in the output column.
    output_label=np.array(output_label)
    # print(output_label.shape)
    label_count = np.zeros(output_label.shape)
    for i in range(0,len(y)):
        for j in range(len(output_label)):
            if y[i] == output_label[j]:
                label_count[j] = label_count[j] + 1
    # print(label_count)
    return label_count

def makeLabelled(column):
    second_limit = column.mean()
    first_limit = 0.5 * second_limit
    third_limit = 1.5*second_limit
    for i in range (0,len(column)):
        if (column[i] < first_limit):
            column[i] = 0
        elif (column[i] < second_limit):
            column[i] = 1
        elif(column[i] < third_limit):
            column[i] = 2
        else:
            column[i] = 3
    return column

def naive_bayes_log(x_train,y_train,x_test):
    # print(x_train.shape)
    type=count_types_output(y_train)
    count=output_counter(y_train)
    prob=count/count.sum()
    for i in prob:
        i=math.log(i,2)
    c=np.zeros(x_test.shape)
    s=0
    x_temp=[[] for a in range(len(type))]
    for i in range(len(type)):
        for j in range(len(y_train)):
            if y_train[j]==type[i]:
                x_temp[i].append(j)
    # print(x_temp)
    prob_val_count=np.zeros(len(type))
    for i in range(len(type)):
        for j in range(len(x_test)):
            for k in range(len(x_temp[i])):
                # print(x_test[j],x_train[x_temp[i][k],j])
                if x_test[j]==x_train[x_temp[i][k],j]:
                    c[j]=c[j]+1
        # print(x_temp[i])
        # print(c)
        for m in range(len(c)):
            c[m]=(c[m]+1)/(count[i]+len(count_types_output(x_train[:,m])))
        for n in range(len(c)):
            s = s + math.log(c[n],2)
        prob_val_count[i] =(prob[i])+s
        for r in range(len(c)):
            c[r]=0
        s=1
        # print(c)
    # print(prob_val_count)
    max_prob = (max(prob_val_count))
    # print(prob_final)
    for i in range(len(prob_val_count)):
        if prob_val_count[i] == max_prob:
            # print(type[i])
            return type[i]


def naive_bayes_normal(x_train,y_train,x_test):
    type=count_types_output(y_train)
    count=output_counter(y_train)
    prob=count/count.sum()
    # print(x_test,count,type,y_train.shape)
    prob_specific_column=np.array(x_test)
    c=0
    s=0
    # print(x_train[4][2])
    prob_final=np.zeros(len(type))
    for i in range(len(type)):
        for k in range(x_test.shape[0]):
            temp_variable = x_test[k]
            c=0
            for j in range(len(y_train)):
                # print(i,j)
                if type[i]==y_train[j]and temp_variable==x_train[j][k] :
                    # print(temp_variable, type[i], y_train[j], x_train[j][k])
                    c=c+1
                else:
                    continue
            # print(c+1,count[i],len(set(x_train[:,k])))
            prob_specific_column[k]=(c+1)/(count[i]+len(set(x_train[:,k])))
            prob_specific_column[k]=math.log(prob_specific_column[k],2)
            # print(prob_specific_column)
            s=0
            for m in prob_specific_column:
                s=m+s
        # print(prob_specific_column.sum(),s)
        prob_final[i]=s

    for i in range(len(prob_final)):
        i=i+math.log(prob[i],2)
    max_prob=(max(prob_final))
    # print(prob_final)
    for i in range(len(prob_final)):
        if prob_final[i]==max_prob:
            # print(type[i])
            return type[i]

def load():
    iris=datasets.load_iris()
    x_train, x_test, y_train, y_test =train_test_split(iris.data,iris.target,test_size=0.5)
    y_pred_1=[]
    y_pred_2 = []
    x_train_1=x_train
    x_test_1=x_test
    for i in range(x_train.shape[1]):
        x_train_1[:,i]=makeLabelled(x_train[:,i])
        x_test_1[:, i] = makeLabelled(x_test[:, i])
    # print(x_train_1)
    for i in range(len(y_test)):
        y_pred_1.append(naive_bayes_normal(x_train_1,y_train,x_test_1[i]))
    y_pred_1=np.array(y_pred_1)
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_test, y_pred_1))
    print(confusion_matrix(y_test, y_pred_1))
    # print(x_train_1)
    for i in range(len(y_test)):
        y_pred_2.append(naive_bayes_log(x_train_1, y_train, x_test_1[i]))
    y_pred_2 = np.array(y_pred_2)
    print(classification_report(y_test, y_pred_2))
    print(confusion_matrix(y_test, y_pred_2))


load()