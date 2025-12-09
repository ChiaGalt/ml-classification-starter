import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report



def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred) #compute confusion matrix from true and predicted labels, because confusion matrix
    #compare the actual target values with those predicted by the model.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names) #here it is used to visualize the confusion matrix.
    
    plt.figure() #to create a new figure
    disp.plot(values_format='d') #to plot the confusion matrix with integer formatting
    plt.title(title) #set the title of the plot
    plt.tight_layout() #tight layout is used to automatically adjust subplot parameters to give specified padding.
    
    if save_path:
        plt.savefig(save_path) #save the plot to the specified path
    else:
        plt.show() #display the plot
        
def print_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names) #classification report is 
    #a text summary of the precision, recall, F1 score for each class in a classification
    print(report) #print the classification report
    