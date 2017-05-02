import graphlab as gl

train_data=gl.SFrame.read_csv('../data/train_FD001.txt.gz',delimiter=' ' ,header=False)
train_data.append(gl.SFrame.read_csv('../data/train_FD002.txt.gz',delimiter=' ' ,header=False))
train_data.append(gl.SFrame.read_csv('../data/train_FD003.txt.gz',delimiter=' ' ,header=False))
train_data.append(gl.SFrame.read_csv('../data/train_FD004.txt.gz',delimiter=' ' ,header=False))
                  
test_data=gl.SFrame.read_csv('../data/test_FD001.txt.gz',delimiter=' ' ,header=False)
test_data.append(gl.SFrame.read_csv('../data/test_FD002.txt.gz',delimiter=' ' ,header=False))
test_data.append(gl.SFrame.read_csv('../data/test_FD003.txt.gz',delimiter=' ' ,header=False))
test_data.append(gl.SFrame.read_csv('../data/test_FD004.txt.gz',delimiter=' ' ,header=False))

Truth=gl.SFrame.read_csv('../data/RUL_FD001.txt.gz',delimiter=' ' ,header=False)
Truth.append(gl.SFrame.read_csv('../data/RUL_FD002.txt.gz',delimiter=' ' ,header=False))
Truth.append(gl.SFrame.read_csv('../data/RUL_FD003.txt.gz',delimiter=' ' ,header=False))
Truth.append(gl.SFrame.read_csv('../data/RUL_FD004.txt.gz',delimiter=' ' ,header=False))


Truth.rename({'X1':'RUL_maxcycle'})   #Rename the only column in the Truth file for test data

#Build the python dictionary
col_renaming_dict={'X1':'id' , 'X2':'cycle'}  
col_renaming_dict.update({'X3':'setting1' , 'X4':'setting2', 'X5':'setting3'})
for i in xrange(1,22):
    #Generates names for columns s1, s2,..., s21
    col_renaming_dict.update({'X'+str(i+5) : 's'+str(i)}) 

#Rename the column in test and data
train_data.rename(col_renaming_dict)
test_data.rename(col_renaming_dict)

#View the train data
train_data.head()


#View the test data
test_data.head()

fault_cycle=train_data.groupby('id',operations={'Fault_cycle' : gl.aggregate.MAX('cycle')})   #Find the last cycle
train_data=train_data.join(fault_cycle) #Add it as a new column to the SFrame
train_data['RUL']=train_data.apply(lambda x: x['Fault_cycle']-x['cycle'])  #Calculate the RUL
#train_data.remove_column('Fault_cycle')  #Optionally remove the column since it won't be needed further
train_data.head() #View the data


Truth['id']=range(1,len(Truth)+1)  #Re-create the engine ID for the data from the "Truth" file. 
test_data=test_data.join(Truth)    #Add this data as a new column to the SFrame

last_test_cycle=test_data.groupby('id',operations={'Last_test_cycle' : gl.aggregate.MAX('cycle')}) #Find the last test cycle for each engine
test_data=test_data.join(last_test_cycle) #Add this as a new column to the SFrame

#The RUL at each cycle is the RUL at the end of the test + how many cycles before end of the test.
test_data['RUL']=test_data.apply(lambda x: x['RUL_maxcycle']+(x['Last_test_cycle']-x['cycle']))

#Optionally remove unnessary columns                                
#test_data.remove_column('RUL_maxcycle')
#test_data.remove_column('Last_test_cycle')

test_data.head() #View the data


w1=30
w0=15

train_data['label1']=train_data.apply(lambda x: 0 if x['RUL']>w1 else 1)
train_data['label2']=train_data.apply(lambda x: x['label1'] if x['RUL']>w0 else 2)
test_data['label1']=test_data.apply(lambda x: 0 if x['RUL']>w1 else 1)
test_data['label2']=test_data.apply(lambda x: x['label1'] if x['RUL']>w0 else 2)



def build_features(data):
    #Defining the Windows onto which we perform the mean and std.
    #Here, the average and std at cycle i will be calculate based on data from cycle i-4 to i. 
    windowsStart = -4
    windowsStop = 0

    #Define the columns for which we want to calculate the mean and std
    cols=list()
    for i in xrange(1,22):
        cols.append('s'+str(i)) 
    col_id_cycle=cols
    col_id_cycle.append('id')
    col_id_cycle.append('cycle')
    
    #Find a list of engine ID
    IDs=data['id'].unique()
    
    CollectFrame=gl.SFrame() # A temporary SFrame for storing temporary data
    for IDx in IDs:  #For each engine
        tmpFrame=data[data['id']==IDx][col_id_cycle].sort('cycle', ascending = True)  #Select all the cycles and sort by cycle ID
        for col in cols:  #Calculate the mean and std for every column of interest
            tmpFrame['mean_'+col]=tmpFrame[col].rolling_mean(windowsStart,windowsStop)
            tmpFrame['stdv_'+col]=tmpFrame[col].rolling_stdv(windowsStart,windowsStop)   
        CollectFrame=CollectFrame.append(tmpFrame)  #Add the new features to the SFrame
    data=data.join(CollectFrame) #When everything has been computed for all engine IDs, join back to the original SFrame. 
    #NB: there might be a more elegant way of doing the above - come back later!
    
    
    #The mean and std, are not defined for the first 4 cycles of each engine. 
    for col in cols:
        #These lines fill the missing data with some reasonable values. 
        data['mean_'+col]=data[{col , 'mean_'+col , 'stdv_'+col}].apply(lambda x :  x[col] if x['mean_'+col] is None  else  x['mean_'+col])    
        data['stdv_'+col]=data[{col , 'mean_'+col , 'stdv_'+col}].apply(lambda x :  0 if x['stdv_'+col] is None  else  x['stdv_'+col])

    return data


#Create the features for the training and test datasets (this can take a while to run)
train_data=build_features(train_data)
test_data=build_features(test_data)


#View the data in 2 different ways
train_data.show() 
train_data.head()


features_to_train=list()
features_to_train.append('cycle')
for i in xrange(1,3):
    features_to_train.append( 'setting'+str(i) )
for i in xrange(1,22):
    features_to_train.append('s'+str(i)) 
    features_to_train.append('mean_s'+str(i))
    features_to_train.append('stdv_s'+str(i))


model=gl.logistic_classifier.create(train_data,'label2',features_to_train,class_weights='auto')


model.evaluate(train_data)

model.evaluate(test_data)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set() 

def gl_confmatrix_2_confmatrix(sf,number_label=3):
    Nlabels=max(len(sf['target_label'].unique()),len(sf['predicted_label'].unique()))
    matrix=np.zeros([number_label,number_label],dtype=np.float)
    for i in sf:
        matrix[i['target_label'],i['predicted_label']]=i['count']
    sum
    
    row_sums = matrix.sum(axis=1) 
    matrix=matrix / row_sums[:, np.newaxis]
    matrix*=100
    
    plt.figure(figsize=(number_label, number_label))
    dims = (8,8)
    fig, ax = plt.subplots(figsize=dims)
    sns.heatmap(matrix, annot=True,  fmt='.2f', xticklabels=['0' ,'1','2'], yticklabels=['0' ,'1','2']);
    plt.title('Confusion Matrix');
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    return matrix


conf_matrix_train=gl.evaluation.confusion_matrix(train_data['label2'],model.predict(train_data))
conf_matrix_test=gl.evaluation.confusion_matrix(test_data['label2'],model.predict(test_data))
gl_confmatrix_2_confmatrix(conf_matrix_train)    
gl_confmatrix_2_confmatrix(conf_matrix_test)


model.coefficients.sort('value').show()


model=gl.logistic_classifier.create(train_data,'label1',features_to_train,class_weights='auto')
conf_matrix_train=gl.evaluation.confusion_matrix(train_data['label1'],model.predict(train_data))
conf_matrix_test=gl.evaluation.confusion_matrix(test_data['label1'],model.predict(test_data))
gl_confmatrix_2_confmatrix(conf_matrix_train,number_label=2)    
gl_confmatrix_2_confmatrix(conf_matrix_test,number_label=2)



model=gl.random_forest_classifier.create(train_data,'label2',features_to_train,class_weights='auto',num_trees=50)
conf_matrix_train=gl.evaluation.confusion_matrix(train_data['label2'],model.predict(train_data))
conf_matrix_test=gl.evaluation.confusion_matrix(test_data['label2'],model.predict(test_data))
gl_confmatrix_2_confmatrix(conf_matrix_train)    
gl_confmatrix_2_confmatrix(conf_matrix_test)



model=gl.boosted_trees_classifier.create(train_data,'label2',features_to_train,class_weights='auto')
conf_matrix_train=gl.evaluation.confusion_matrix(train_data['label2'],model.predict(train_data))
conf_matrix_test=gl.evaluation.confusion_matrix(test_data['label2'],model.predict(test_data))
gl_confmatrix_2_confmatrix(conf_matrix_train)    
gl_confmatrix_2_confmatrix(conf_matrix_test)