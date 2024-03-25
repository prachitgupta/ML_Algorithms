##ref := https://docs.streamlit.io/library/api-reference

import streamlit as st
import pandas as pd
import types
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class ProcessedData:
    def __init__(self,data,targetName):
        self.targetName = targetName
        self.data = data
        self.N = data.shape[0] 
        self.D = data.shape[1] - 1 ##axis 1 
        self.features = data.drop(targetName, axis=1)##quality = target selectds all columns expect quality
        self.target = data[targetName]
        self.description = data.describe()
        self.NullCount= data.isnull().sum()
        
    def count_outliers(self, columnName):
        df = self.data
        Q1 = df[columnName].quantile(0.25) # first quartile of data
        Q3 = df[columnName].quantile(0.75)# rnd quartile 
        IQR = Q3 - Q1  #3 range we are assuming to contain data (excluding outliers)
        lower_bound = Q1 - 1.5 * IQR ##chat gpt selected appropriate limits below which we can easily say data is outlier
        upper_bound = Q3 + 1.5 * IQR
        df_removedOutliers = (df[columnName] >= lower_bound) & (df[columnName] <= upper_bound)
        self.OutliersCount = df_removedOutliers.shape - df_removedOutliers.sum() ##only true i.e 1 counted in sum
        return OutliersCount
    
    def remove_outliers(self, column):
        df = self.data
        Q1 = df[column].quantile(0.25) # first quartile of data
        Q3 = df[column].quantile(0.75)# rnd quartile 
        IQR = Q3 - Q1  #3 range we are assuming to contain data (excluding outliers)
        lower_bound = Q1 - 1.5 * IQR ##chat gpt selected appropriate limits below which we can easily say data is outlier
        upper_bound = Q3 + 1.5 * IQR
        median = df[column].median()
        df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
        self.data = df
        
    def impute_data(self):
        ##select feautres to impute
        features_for_imputation = self.data.select_dtypes(include='number').columns ##all features are numeric for classification
        # Initialize IterativeImputer
        imputer = IterativeImputer(random_state=0)
        # Fit and transform the data using IterativeImputer ref : chat gpt 
        data_imputed = pd.DataFrame(imputer.fit_transform(self.data[features_for_imputation]), columns=features_for_imputation)
        ##transformed data
        self.data[features_for_imputation] = data_imputed

    def trainTestSplit(self):
        ##X = FEATURE AND Y =TARGET, of new proccedData which is stored in self.data
        ##ref gpt  both datasets only numeric feautures
        numeric_features = self.data.select_dtypes(include='number').columns #3only select number columns
        # Select features with numeric data
        X = self.data[numeric_features]
        ##in regression target column also numeric so
        if self.targetName in X.columns:
            X.drop(self.targetName, axis=1, inplace=True)
        y = self.data[self.targetName]
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) ##70 % for training
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def standardize(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train) 
        self.X_val_scaled = scaler.transform(self.X_val)
        self.X_test_scaled = scaler.transform(self.X_test)
    

class model:
    def __init__(self,ProcessedData,problem): ##pass object of type ProcessedData (have all member functions,variables)
        self.data = ProcessedData
        ##Processed data must have been splited in testing and training set a priori
        self.X_train_scaled = ProcessedData.X_train_scaled  ##we need training data for this class
        self.y_train = ProcessedData.y_train
        self.is_regression = problem == "regression"
    # 1. Random Forest  (creates multiple decesion tress and average out the prediction value of each tree)
    ##ref chat gpt and  scikit_learn RandomForestRegressor
    
    def randomForest(self):        
        ## initialize with initial random value 42(chat gpt) reproducibility
        rf_model = RandomForestRegressor(random_state=42) if self.is_regression else RandomForestClassifier(random_state=42, max_depth=None) 
        Paramgrid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}##hyperparameters varying = no of trees, depth of trees
        ##use GridSearchCV to minimize mean square error , and find best hyperparams to fit model on training data
        grid_rf = GridSearchCV(rf_model, Paramgrid_rf, cv=3, scoring='neg_mean_squared_error' if self.is_regression else 'accuracy')## 3 fold verification
        grid_rf.fit(self.X_train_scaled, self.y_train)##fit model
        n_estimators, max_depth =  grid_rf.best_params_ #3optimum hyperparams
        ##corresponding best model
        Best_rf_model = grid_rf.best_estimator_
        return Best_rf_model

  
        

# Load the dataset
wine_red = pd.read_csv("winequality-red.csv", sep=';')
## create instance of class ProcessData
Red = ProcessedData(wine_red,"quality")
##remove outliers in target
Red.remove_outliers("quality")
##fill null points
Red.impute_data()
##split 
Red.trainTestSplit()
##finally standardise
Red.standardize()

##now our datasets have been processed and splitted into training test and validation 
##train the three models
ModelRed = model(Red,"regression") ##instance of model class with different data
##final model to deploy on frontend
trained_model = ModelRed.randomForest()
# Define GUI elements
st.title('Wine Quality Predictor')
st.sidebar.header('Input Parameters') ##sidebars

##input features to display (min max arbitary from prelimanary inspection of data)
acidity = st.sidebar.slider('Acidity', min_value=0.0, max_value=10.0, value=5.0)
fixed_acidity = st.sidebar.slider('Fixed Acidity', min_value=0.0, max_value=15.0, value=7.0)
volatile_acidity = st.sidebar.slider('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.5)
citric_acid = st.sidebar.slider('Citric Acid', min_value=0.0, max_value=1.0, value=0.5)
residual_sugar = st.sidebar.slider('Residual Sugar', min_value=0.0, max_value=30.0, value=15.0)
chlorides = st.sidebar.slider('Chlorides', min_value=0.0, max_value=1.0, value=0.08)
free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', min_value=0, max_value=100, value=30)
total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', min_value=0, max_value=300, value=150)
density = st.sidebar.slider('Density', min_value=0.0, max_value=2.0, value=1.0)
pH = st.sidebar.slider('pH', min_value=0.0, max_value=14.0, value=3.0)
sulphates = st.sidebar.slider('Sulphates', min_value=0.0, max_value=2.0, value=0.5)
alcohol = st.sidebar.slider('Alcohol', min_value=8.0, max_value=15.0, value=10.0)

# Define prediction function
def predict_wine_quality(input_features, model):
    # Make prediction using the model
    prediction = model.predict([input_features]) ##prediction over all input features
    return prediction


if st.button('Predict'):# if button pressed
    input_features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                      free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
    quality = predict_wine_quality(input_features, trained_model) ##final prediction
    st.write(f'Predicted Wine Quality: {quality}')

##some additional Gui found in doucumentation
histogram = st.sidebar.checkbox('Plot Feature Histogram', value=False,
                                      help="Toggle to plot histogram of selected features.")
if histogram:
    importances = trained_model.feature_importances_ 
    st.title('Feature Importance')
    st.subheader('Random Forest')
    plt.figure(figsize=(10, 5))
    plt.bar(wine_red.columns[:-1], importances, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Display plot in Streamlit app
    ##disable deprication warning
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()