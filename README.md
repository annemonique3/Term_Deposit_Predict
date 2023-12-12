# Point
author: Anne Uwamahoro
auwamahoro@westmont.edu

#Term_Deposit_Prediction

#Introduction: 

This program uses the Guassian Naive Bayes Technique to predict if a customer will make term deposit or not. The classifier is locally built in the program, trained using the train data and tested using the test data provided in the code. 
Guassian Naive Bayes was chosen to be used in this program because it is suitable for data type in the code. It also uses probability to make predictions which exactly suits what I am seeking to achieve. It is also good with new data because it facilitates easy updates. 
Additionally, since I am working with binary description on term description(y), Guassian Naive Bayes is well suited for this. 

#Data Preprocessing: 

Since we have both numerical and categorical variables, there needs to be standardizing of the data set variables. StandardScaler is used to standardize numerical features while OneHotEncorder standardizes the categorical features. The importance of this is so that the model treats all features equally to prevent any bias in the prediction. 

#Model Implementation:

This part mainly focuses on creating a classifier that determines who will and will not subscribe to the term deposit. 
The model class starts with the init method to initialize crucial properties of the classifier. It also has the fit method which fits the model to training data. This method also initializes the arrays of mean, variance, and prior probabilities of each class. 
Next, is the predict method that makes predictions on the data. The pdf method further calculates the probability density function for a feature  and returns it. 

After a model is created, it is now trained and evaluated with its accuracy and classification report printed.

#Resources used: 
- Kaggle.com
- Numpy.org
- Chatgpt
  Prompt: How can a feature of class be computed: 
  mean = self.mean[class_idx]
  var = self.var[class_idx]
  numerator = np.exp(- (x - mean) ** 2 / (2 * var))
  denominator = np.sqrt(2 * np.pi * var)
  return numerator / denominator
