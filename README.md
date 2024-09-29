

As part of my internship with Oasis Infobyte, I was entrusted with three key machine learning projects, each addressing unique business needs across different domains. These projects provided an excellent opportunity to apply advanced data science techniques to real-world problems, enhancing my practical expertise in the field. The tasks I completed were as follows:

## TASK 1 - Iris Flower Classification

This project implements a machine learning model to classify iris flowers into three species: Setosa (Iris-setosa), Versicolor (Iris-versicolor), and Virginica (Iris-virginica). The dataset used is the famous Iris dataset, which consists of 150 samples with four features: sepal length, sepal width, petal length, and petal width. The objective is to design a classifier that can accurately determine the type of iris flowers based on these characteristics.

### Features:
- **Sepal Length:** Length of the sepal in centimeters.
- **Sepal Width:** Width of the sepal in centimeters.
- **Petal Length:** Length of the petal in centimeters.
- **Petal Width:** Width of the petal in centimeters.

### Project Structure:
1. **Data Preprocessing:** This step involves cleaning and normalizing the dataset to prepare it for model training.
2. **Model Training:** Various machine learning algorithms, including Naive Bayes, SVM, Random Forest, Logistic Regression, K-Nearest Neighbors, and Decision Trees, are utilized to classify the iris species.
3. **Model Evaluation:** The models are assessed for accuracy and performance using relevant metrics.
4. **Web Interface:** A simple web interface is developed using Gradio, allowing users to input flower measurements and obtain real-time species predictions.

### Best Performing Model:
The best performing model is **K-Nearest Neighbors**, achieving a precision of 1.0.

---

## TASK 2 - Gmail Spam Detection Using Machine Learning

This project focuses on categorizing incoming emails as spam or not spam (ham) using machine learning. By applying natural language processing (NLP) techniques, the model analyzes the content of messages and calculates the probabilities of them being classified as spam. The goal is to enhance email filtering systems and help users avoid unwanted or potentially dangerous messages.

### Features:
- **Email:** The content of the email, which includes both the subject and body text.
- **Label:** Classifies the email as either spam or not spam.

### Project Structure:
1. **Data Preprocessing:** This includes tokenizing, cleaning, and vectorizing email data by removing punctuation, converting text to lowercase, and eliminating stopwords.
2. **Feature Extraction:** Techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) are employed to convert email text into numerical features for analysis.
3. **Model Training:** Various machine learning models, including Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, and a Voting Classifier, are trained to classify the emails.
4. **Model Evaluation:** The performance of the models is assessed using metrics like accuracy and precision.
5. **Web Interface:** A simple web application is built using Streamlit, enabling users to input email content and receive predictions on whether the email is spam or not.

### Best Performing Model:
The best performing model is **Multinomial Naive Bayes**, which effectively classifies emails with high accuracy.

---

## TASK 3 - Sales Prediction Using Machine Learning

This project demonstrates the use of multiple regression techniques to predict sales based on advertising data from TV, Radio, and Newspaper channels. Various models, including linear regression, decision trees, and random forests, are evaluated to determine which model best predicts sales based on the dataset provided.


### Features:
- **TV**: Advertising budget for TV.
- **Radio**: Advertising budget for Radio.
- **Newspaper**: Advertising budget for Newspaper.
- **Sales**: Target variable representing sales generated.

### Project Structure:
1. **Data Collection:** Gathering historical sales data along with related features from various sources.
2. **Data Preprocessing:** Cleaning the dataset, addressing missing values, and encoding categorical variables for effective analysis.
3. **Model Training:** Building machine learning models using techniques such as Linear Regression,Decision Tree and Random Forest for predicting sales.
4. **Model Evaluation:** Evaluating model performance with metrics such as Mean Square Error.
5. **Forecasting Interface:** Developing a user-friendly interface using Streamlit that allows users to input parameters and receive sales forecasts.

### Best Performing Model:
The best performing model is **Linear Regression**, which demonstrates the lowest Mean sqaured Percentage Error in sales predictions.

---

## Technologies Used
- Python
- Scikit-learn for model building
- Gradio and Streamlit for web interfaces
- Pandas and NumPy for data manipulation
- Matplotlib or Seaborn for data visualization
- NLTK for Natural Language Processing
