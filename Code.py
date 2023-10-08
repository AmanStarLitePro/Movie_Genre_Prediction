import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# Loading Data from Churn_Modelling.csv in pandas DataFrame
# Since Data is not in the form of csv UTF-8 Format, we convert it into ISO-8859-1 format
train_df = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['Title', 'Genre', 'Description'], nrows=6000)
test_df = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['Title', 'Description'], nrows=6000)
test_data_solution_df = pd.read_csv('test_data_solution.txt', sep=' ::: ', engine='python', names=['Title', 'Genre', 'Description'], nrows=6000)

# Define the text columns you want to convert to TF-IDF
text_columns = ['Title', 'Description']

# Create TF-IDF vectorizers for each text column
tfidf_vectorizers = {col: TfidfVectorizer() for col in text_columns}

# Initialize a list to store TF-IDF data
X_train_tfidf = []
X_test_tfidf = []

# Iterate through text columns for TF-IDF transformation
for col in text_columns:
    try:
        tfidf_data = tfidf_vectorizers[col].fit_transform(train_df[col])
        X_train_tfidf.append(tfidf_data)

        # Check if the column exists in the test_df before transforming
        if col in test_df.columns:
            tfidf_data_test = tfidf_vectorizers[col].transform(test_df[col])
            X_test_tfidf.append(tfidf_data_test)
    except ValueError as e:
        print(f"Skipping column '{col}' due to error: {e}")

# Combine TF-IDF features into a single sparse matrix
X_train_final = hstack(X_train_tfidf)
X_test_final = hstack(X_test_tfidf)

# Encode the 'Genre' column using LabelEncoder
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(train_df['Genre'])
Y_test = label_encoder.transform(test_data_solution_df['Genre'])

# Training the models
model1 = LogisticRegression(solver='liblinear', max_iter=1000)
model2 = SVC()
model3 = MultinomialNB()

# Training the Logistic Regression Model, SVC Model, Naive Bayes MultiNomialNB Model with the training data
model1.fit(X_train_final, Y_train)
model2.fit(X_train_final, Y_train)
model3.fit(X_train_final, Y_train)

# For Model1 LogisticRegression
# Prediction on training data
Prediction_on_Train_data = model1.predict(X_train_final)
accuracy_on_Train_data = accuracy_score(Y_test, Prediction_on_Train_data)
print("The accuracy score on Test Data using Logistic Regression Model is : " + str(accuracy_on_Train_data * 100) + "%")

# Prediction on testing data
Prediction_on_Test_data = model1.predict(X_test_final)
accuracy_on_Test_data = accuracy_score(Y_test, Prediction_on_Test_data)
print("The accuracy score on Test Data using Logistic Regression Model is : " + str(accuracy_on_Test_data * 100) + "%")

# For Model2 SVC
# Prediction on training data
Prediction_on_Train_data = model2.predict(X_train_final)
accuracy_on_Train_data = accuracy_score(Y_test, Prediction_on_Train_data)
print("The accuracy score on Test Data using SVC Model is : " + str(accuracy_on_Train_data * 100) + "%")

# Prediction on testing data
Prediction_on_Test_data = model2.predict(X_test_final)
accuracy_on_Test_data = accuracy_score(Y_test, Prediction_on_Test_data)
print("The accuracy score on Test Data using SVC Model is : " + str(accuracy_on_Test_data * 100) + "%")

# For Model3 Naive Bayes MultinomialNB
# Prediction on training data
Prediction_on_Train_data = model3.predict(X_train_final)
accuracy_on_Train_data = accuracy_score(Y_test, Prediction_on_Train_data)
print("The accuracy score on Test Data using Naive Bayes MultiNomialNB is : " + str(accuracy_on_Train_data * 100) + "%")

# Prediction on testing data
Prediction_on_Test_data = model3.predict(X_test_final)
accuracy_on_Test_data = accuracy_score(Y_test, Prediction_on_Test_data)
print("The accuracy score on Test Data using Naive Bayes MultiNomialNB is : " + str(accuracy_on_Test_data * 100) + "%")

# Custom Input for Prediction
custom_input = {'Title': 'El enfermo imaginario (2011)', 'Description': "Argan, a rich, eccentric hypochondriac, will do anything to defeat his fear of dying. When he tries to marry daughter Angelique to doctor Cleonte, Argan's rebellious daughter, strong-willed servants, and scheming wife all swing into action to save themselves from the impending domestic disaster. Can anyone reform Argan's health care? Can his brother convince him he's not really dying?"}

# Transform the custom input using the TF-IDF vectorizers
custom_input_transformed = [tfidf_vectorizers[col].transform([custom_input[col]]) for col in text_columns]

# Combine the transformed custom input into a single sparse matrix
custom_input_final = hstack(custom_input_transformed)

# Predict the genre for the custom input
custom_input_prediction1 = model1.predict(custom_input_final)
custom_input_prediction2 = model2.predict(custom_input_final)
custom_input_prediction3 = model3.predict(custom_input_final)


# Decode the predicted genre using the label encoder
predicted_genre = label_encoder.inverse_transform(custom_input_prediction1)
print("Predicted Genre for Custom Input using Logistic Regression: " + predicted_genre[0])

# Decode the predicted genre using the label encoder
predicted_genre = label_encoder.inverse_transform(custom_input_prediction2)
print("Predicted Genre for Custom Input using SVC Model: " + predicted_genre[0])

# Decode the predicted genre using the label encoder
predicted_genre = label_encoder.inverse_transform(custom_input_prediction3)
print("Predicted Genre for Custom Input using Naive Bayes MultiNomialNB: " + predicted_genre[0])
