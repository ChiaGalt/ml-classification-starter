import os 

from sklearn.datasets import load_iris #importing the iris dataset 
from sklearn.linear_model import LogisticRegression #importing logistic regression model
from sklearn.model_selection import train_test_split #importing train test split function
from sklearn.neighbors import KNeighborsClassifier #importing k-nearest neighbors model
from sklearn.preprocessing import StandardScaler #importing standard scaler for feature scaling, used 
#to standardize features by removing the mean and scaling to unit variance. in this way, each feature
#will have the properties of a standard normal distribution with a mean of 0 and a standard deviation of 1. 
#we remove the mean and scale to unit variance because many machine learning algorithms perform better or converge
#faster when features are on a relatively similar scale and close to normally distributed. 
#the std normal distribution is used as a reference point for this scaling because of its well-known properties.
from sklearn.svm import SVC #importing support vector classifier model

from utils import plot_confusion_matrix, print_classification_report 

def ensure_output_dir(path="outputs"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_data(test_size=0.3, random_state=42): #dft values, test_size means that 30% of the data will be used for testing, 
    #while random_state means that the data will be split in a reproducible way. we use random_state to ensure that 
    #we get the same split every time we run the code. this means that the results of our model training and 
    #evaluation will be consistent across different runs.
    
    iris = load_iris() #loading the iris dataset
    X=iris.data #features, namely sepal length, sepal width, petal length, petal width
    y=iris.target #target labels
    class_names=iris.target_names #class names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    #here x_train and x_test are the feature sets for training and testing respectively, 
    #while y_train and y_test are the corresponding target labels. target labels are the species of iris flowers.
    #stratify=y means that the split will maintain the same proportion of each class in both the training and testing sets.
    #we say equal to y because y contains the class labels.
    return X_train, X_test, y_train, y_test, class_names

def train_logreg(X_train, y_train): #function to train logistic regression model
    model = LogisticRegression(max_iter=100) #max_iter is the maximum number of iterations taken for the solvers to converge
    model.fit(X_train, y_train) #fitting the model to the training data. fitting means that the model learns the relationship 
    #between the features and the target labels.
    return model

def train_svm(X_train, y_train): #function to train support vector machine model
    model = SVC(kernel="rbf") #svc is the support vector classifier, kernel is the type 
    #of kernel to be used in the algorithm. the kernel is a function that takes the input data 
    #and transforms it into a higher-dimensional space. rbf stands for radial basis function, which is a popular kernel choice for SVMs.
    #radial basis function means that the function's value depends on the distance from a central point.
    model.fit(X_train, y_train) #fitting the model to the training data
    return model

def train_knn(X_train, y_train, n_neighbors=3): #function to train k-nearest neighbors model
    #the k-nearest neighbors algorithm is a supervised machine learning algorithm used for classification and 
    #regression. in both cases, the algorithm works by finding the k-nearest data points in the training dataset 
    #to a given input data point and using those neighbors to make a prediction. in this case, we are using it 
    #for classification.
    model = KNeighborsClassifier(n_neighbors=5) #n_neighbors is the number of neighbors to use, 
    #this value determines how many nearby data points will be considered when making a prediction for a new data point
    #and we choose it based on the dataset and the problem at hand. in this case we choose 5
    model.fit(X_train, y_train) #fitting the model to the training data
    return model

def main():
    output_dir = ensure_output_dir("outputs") #ensuring that the output directory exists
    X_train, X_test, y_train, y_test, class_names = load_data() #loading the data
    
    scaler = StandardScaler() #initializing the standard scaler
    X_train_scaled = scaler.fit_transform(X_train) #fitting and transforming the training data using the scaler
    #fit_transform means that the scaler learns the mean and standard deviation from the training data
    #and then applies the scaling to the training data. x train scaled is the scaled version of x train
    X_test_scaled = scaler.transform(X_test) #transforming the test data using the scaler
    #here we only transform the test data using the mean and standard deviation learned from the training data.
    #the difference from x train scaled is that we do not fit the scaler again on the test data because we want 
    #to use the same scaling as the training data.
    models = {
        "Logistic Regression": train_logreg(X_train_scaled, y_train),
        "Support Vector Machine": train_svm(X_train_scaled, y_train),
        "K-Nearest Neighbors": train_knn(X_train_scaled, y_train),
    }
    #the scaler is applied only on the features, not on the target labels, because the target labels are 
    #categorical values representing the species of iris flowers.
    for name, model in models.items(): #items are the pairs of name of the model and model from the functions above
        print(f"\n==={name.upper()}===") #printing the name of the model in uppercase
        y_pred = model.predict(X_test_scaled) #model.predict is used to make predictions on the test data
        #this function takes the test features as input and outputs the predicted class labels. y pred is the predicted labels
        #.predict is used after the model has been trained using the training data, as we did durint the creation of the dictionary
        print_classification_report(y_test, y_pred, class_names) #printing the classification report
        
        save_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred, class_names, save_path=save_path, title=f"{name} Confusion Matrix")
        print(f"Confusion matrix saved to {save_path}")
    #we generate and save confusion matrices for each model, using the true labels and predicted labels.
    
if __name__ == "__main__": 
    main()