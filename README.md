E-commerce Purchase Intent Prediction
Project Overview
This project uses Logistic Regression to predict whether an e-commerce website session will result in a purchase.
The objective is to build a machine learning classification model that can identify purchase intent based on user session behavior.

The project covers the complete machine learning workflow, including:
- Dataset inspection
- Data quality checking
- Exploratory analysis
- Data preprocessing
- Train/test splitting
- Logistic Regression model training
- Model evaluation
- Coefficient interpretation
- Prediction on a new website session
- Discussion of limitations and possible improvements

Dataset
Dataset: "dataset_04_ecommerce_purchase_intent.csv"
The dataset contains 1,000 website sessions and 7 columns:

Input Features
Feature| Description
"pages_viewed"| Number of pages viewed during the session
"session_minutes"| Duration of the website session in minutes
"products_viewed"| Number of products viewed
"cart_additions"| Number of products added to the shopping cart
"discount_seen"| Number of discount-related exposures/interactions
"previous_orders"| Number of previous orders made by the customer

Target Variable
Target| Meaning
"0"| No purchase
"1"| Purchase
The target variable is therefore a binary classification problem.

Data Quality
The dataset was inspected before model training.
Results
- Total observations: 1,000
- Total variables: 7
- Missing values: 0
- Duplicate rows: 0
- Target classes: 500 no-purchase / 500 purchase
- Class distribution: 50% / 50%

Because the target classes are perfectly balanced, no class-balancing technique such as oversampling or undersampling was required.
All predictor variables are numerical, so categorical encoding was not necessary.

Machine Learning Approach
1. Feature and Target Separation
The target variable "target" was separated from the predictor variables.

X = df.drop("target", axis=1)
y = df["target"]


2. Train/Test Split
The dataset was divided into:
- 80% training data: 800 sessions
- 20% testing data: 200 sessions

A stratified split was used to preserve the 50/50 class distribution in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
The test data was kept separate from the training process so that it could provide an unbiased evaluation of the final model.

3. Feature Scaling
"StandardScaler" was used to standardize the numerical features.
Scaling was performed inside a Pipeline, ensuring that the scaler was fitted only on the training data.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
Using a pipeline helps prevent data leakage because information from the test set is not used when fitting the scaler.

Logistic Regression Model
The required machine learning algorithm for this project is Logistic Regression.
The model estimates the probability that a website session will result in a purchase.

The main configuration was:
Algorithm: Logistic Regression
Scaling: StandardScaler
Maximum iterations: 1000
Random state: 42
Train/test split: 80/20
Stratification: Yes

The model was trained using:
model.fit(X_train, y_train)

Predictions and purchase probabilities were then generated:
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

Model Evaluation
The model was evaluated using the required classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Results
Metric| Result
Accuracy| 73.00%
Precision| 73.96%
Recall| 71.00%
F1-score| 72.45%
ROC-AUC| 80.05%


Confusion Matrix
                         Predicted
                         No    Yes
Actual No        75    25
Actual Yes       29    71

Therefore:
- True Negatives (TN): 75
- False Positives (FP): 25
- False Negatives (FN): 29
- True Positives (TP): 71
The model correctly classified 146 out of 200 test sessions, resulting in an accuracy of 73%.
Interpretation of Results
The model achieved an accuracy of 73% on the held-out test set.
The ROC-AUC of 0.8005 indicates that the model has reasonably good ability to distinguish between sessions that result in a purchase and those that do not.
The recall of 71% means that the model correctly identified approximately 71% of the actual purchasing sessions in the test set.
The precision of 73.96% means that when the model predicted a purchase, approximately 74% of those predictions were actually purchases.
The F1-score of 72.45% provides a balance between precision and recall.

Feature Interpretation
Because the Logistic Regression model was trained with standardized features, the coefficients indicate the direction and relative strength of the relationship between each feature and purchase probability.

Model Coefficients
Feature| Coefficient| Relationship
"discount_seen"| +0.6525| Positive
"pages_viewed"| +0.6367| Positive
"products_viewed"| +0.6121| Positive
"session_minutes"| -0.3366| Negative
"cart_additions"| -0.3414| Negative
"previous_orders"| -0.3943| Negative

Interpretation
The three strongest positive coefficients were:
1. "discount_seen"
2. "pages_viewed"
3. "products_viewed"
This indicates that, within this dataset and while holding the other variables constant, higher values of these features are associated with higher predicted purchase probability.
The largest negative coefficient was associated with "previous_orders".
It is important to note that these relationships are associations rather than proof of causation. The model does not establish that increasing or decreasing one of these variables will directly cause a customer to purchase.

Example Prediction
A new website session can be passed to the trained model for prediction.

Example:
new_session = pd.DataFrame({
    "pages_viewed": [8],
    "session_minutes": [12],
    "products_viewed": [5],
    "cart_additions": [2],
    "discount_seen": [3],
    "previous_orders": [1]
})

prediction = model.predict(new_session)
probability = model.predict_proba(new_session)[:, 1]



For this example session, the model predicts:
Prediction: Purchase
Purchase probability: 57.73%
Therefore, based on the trained Logistic Regression model, this particular session is classified as likely to result in a purchase.

Project Structure
A recommended project structure is:
Problem_4_Ecommerce_Purchase_Intent/
│
├── dataset_04_ecommerce_purchase_intent.csv
├── Problem_4_Ecommerce_Purchase_Intent.ipynb
├── Problem_4_Ecommerce_Purchase_Intent_Project_Report.docx
├── Problem_4_Ecommerce_Purchase_Intent_Project_Workbook.xlsx
└── README.md

Technologies Used
The project was implemented using Python and common machine learning/data analysis libraries.

Libraries
pandas
numpy
scikit-learn
matplotlib
seaborn

Main Scikit-learn Components
train_test_split
StandardScaler
Pipeline
LogisticRegression
accuracy_score
precision_score
recall_score
f1_score
roc_auc_score
confusion_matrix
classification_report

How to Run the Project
1. Install the required libraries
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

2. Open the notebook
jupyter notebook

3. Load the dataset
Place:
dataset_04_ecommerce_purchase_intent.csv
in the same directory as the notebook.

4. Run the notebook
Execute the cells in order to:
1. Load the dataset
2. Inspect the data
3. Check data quality
4. Explore the target variable
5. Split the dataset
6. Scale the features
7. Train Logistic Regression
8. Generate predictions
9. Calculate evaluation metrics
10. Analyze coefficients
11. Predict the outcome of a new session

Limitations
Although the model achieved reasonable performance, several limitations should be considered.

1. Dataset Size
The dataset contains only 1,000 observations, which is relatively small for a production e-commerce prediction system.

2. Limited Features
Only six predictors are available. Real e-commerce systems may use many additional variables, such as:
- Device type
- Traffic source
- Product price
- Customer demographics
- Geographic location
- Time of day
- Seasonality
- Previous browsing behavior
- Product category
- Marketing campaign
- Customer lifetime value

3. Generalization
The reported performance is based on a single held-out test split. Performance on future or different customer populations may be different.
4. Behavioral Changes
Customer behavior can change over time because of changes in pricing, promotions, products, competitors, or economic conditions.
5. Correlation Does Not Imply Causation
The model coefficients describe statistical relationships in the dataset. They should not be interpreted as proof that a feature directly causes a purchase.

Possible Improvements
Future versions of the project could improve the model by:
- Collecting a larger dataset
- Adding more relevant behavioral features
- Using temporal validation with future sessions
- Performing cross-validation
- Hyperparameter tuning
- Engineering interaction and nonlinear features
- Optimizing the prediction threshold according to the business objective
- Monitoring model performance after deployment
- Periodically retraining the model using newer data

Conclusion
This project developed a Logistic Regression model for predicting e-commerce purchase intent.

The final model achieved:
73.00% accuracy and 0.8005 ROC-AUC on the held-out test set.
The results demonstrate that session-level behavioral features can provide useful information for distinguishing between purchasing and non-purchasing website sessions.
The Logistic Regression model was selected as the final model because it satisfies the project requirements, provides reasonable predictive performance, and is relatively easy to interpret. Its coefficients also provide insight into which features are positively or negatively associated with purchase probability.
Overall, the project demonstrates a complete and reproducible machine learning workflow, from data inspection and preprocessing through model training, evaluation, interpretation, and prediction.
