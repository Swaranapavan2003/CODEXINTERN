# House Price Prediction Project
# House Price Prediction using California Housing Dataset
        This project is a simple implementation of a House Price Prediction model using the California Housing dataset from sklearn. It demonstrates how to load, preprocess, and train a regression model to predict housing prices based on features such as size, location, number of bedrooms, and more.

#Objective: 
      Predict the price of a house based on features such as size, location, and number of bedrooms using machine learning.

#Dataset:
  California Housing Dataset
      *Loaded using sklearn.datasets.fetch_california_housing. It contains:
          * 20,640 samples
          * 8 numerical features (e.g., MedInc, HouseAge, AveRooms, etc.)
          Target: Median house value for California districts

#Step 1: Import Required Libraries
        Essential Python libraries such as Pandas, NumPy, Scikit-learn, Seaborn, and Matplotlib are imported for data                 manipulation, visualization, modeling, and evaluation.

#Step 2: Load and Explore the Dataset
        * The Boston Housing dataset is loaded and converted into a DataFrame.
        * Initial exploration is performed using '.head()' and '.info()'.

#Step 3: Visualize Data
        * Distribution of house prices is plotted using a histogram.        * A correlation heatmap shows how each feature correlates with price.

 #Step 4: Check for Missing Values
        * The dataset is scanned for any missing values.

 #Step 5: Preprocessing (Feature Scaling)
        *Feature data is standardized using 'StandardScaler'.

 #Step 6: Split Data

        * Dataset is split into **80% training** and **20% testing** using             'train_test_split()'.

#Step 7: Model Training
        * A **Linear Regression** model is trained on the training set.

#Step 8: Predictions
        * The trained model makes predictions on the test set.

 #Step 9: Model Evaluation
      * Metrics like **Mean Squared Error (MSE)** and **R-squared (R²)**              score are computed.

#Step 10: Visualization

       * A scatter plot is created to visualize "actual vs predicted                  prices".

#Sample Output:
         Mean Squared Error (MSE): 24.29
         R-squared (R^2 Score): 0.71

#Skills Gained

      * Data visualization and exploration
      * Regression modeling using scikit-learn
      * Model evaluation using MSE and R²
      * Feature scaling and data preprocessing

#Future Enhancements

      * Use other regression models like Random Forest or XGBoost
      * Apply hyperparameter tuning (GridSearchCV)
      * Replace the dataset with a modern housing dataset (e.g., from Kaggle)

 #Author"

    Swarna Pavan
          AI/ML Intern | CodexIntern

#Note:
       As of scikit-learn v1.2, the Boston dataset is deprecated due to ethical concerns about the B feature. You can use  California Housing Dataset as a modern alternative..

#License
    This project is open-source and available for educational purposes.

