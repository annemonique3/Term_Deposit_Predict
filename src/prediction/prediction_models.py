# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

__author__ = "Anne Uwamahoro"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Anne Uwamahoro"]
__license__ = "MIT"
__email__ = "auwamahoro@westmont.edu"

# Data Preprocessing
# Loading training data
train_df = pd.read_csv('/Users/CSUFTitan/Downloads/archive/train.csv')

# Loading test data
test_df = pd.read_csv('/Users/CSUFTitan/Downloads/archive/test.csv')

# Identifying categorical columns and removing the target variable
category_columns = train_df.select_dtypes(include=['object']).columns.tolist()
category_columns.remove('y')

# Identifying numerical columns
numerical_columns = train_df.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), category_columns)
    ]
)

# Preprocessing the features of training and testing data
X_train = preprocessor.fit_transform(train_df.drop('y', axis=1))
X_test = preprocessor.transform(test_df.drop('y', axis=1))

# Converting the target variable in training and testing data into binary
y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values
y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values


# Building the Gaussian Naive Bayes Model
class TermDepositPrediction:
    # Initializing the properties of the classifier
    def __init__(self):
        # To store unique class labels
        self.classes = None

        # To store the mean of features for each class
        self.mean = None

        # To store the variance of features for each class
        self.var = None

        # To store the prior probabilities of each class
        self.priors = None

    def fit(self, X, y):
        # Fitting the model to training data
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initializing arrays for mean, variance, and priors for each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        # Iterating over each class
        for clas in self.classes:
            x_c = X[y == clas]
            self.mean[clas, :] = x_c.mean(axis=0)
            self.var[clas, :] = x_c.var(axis=0)
            self.priors[clas] = x_c.shape[0] / float(n_samples)

    def predict(self, X):
        # Making predictions on data
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Helper method to calculate log posterior probability for each class
        posteriors = []

        # Iteration over each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # Compute the probability density function of a feature
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Training the Model
model = TermDepositPrediction()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
