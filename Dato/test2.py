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