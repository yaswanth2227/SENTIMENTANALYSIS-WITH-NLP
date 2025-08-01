# SENTIMENTANALYSIS-WITH-NLP

COMPANY : CODTECH IT SOLUTIONS

NAME : B.Yaswanth Kumar

INTERN ID : CT06DZ363

DOMAIN : MACHINE LEARNING

DURATION : 6 WEEKS

MENTOR : NEELA SANTHOSH

**This task involves developing a sentiment analysis model that processes customer reviews and classifies them as either positive or negative.

The objective is to implement the entire workflow using only pure Python, without relying on any external machine learning or natural language processing

libraries such as scikit-learn, pandas, or numpy. The goal is to gain a deeper understanding of the inner workings of TF-IDF vectorization and logistic regression by constructing them manually.

Dataset and Preprocessing: The dataset used in this task consists of a small number of manually defined customer reviews.

Each review is paired with a sentiment label: 1 for positive sentiment and 0 for negative sentiment.

Example positive reviews include phrases like "I love this product" or "This is the best thing ever", while negative reviews include statements such as "I hate this" or "This is awful and terrible".

The preprocessing begins by tokenizing each review, which involves converting the text to lowercase and splitting it into individual words. A vocabulary is then constructed by collecting all unique words across the dataset. Each word is assigned an index, creating a mapping that is used to compute word frequencies and TF-IDF vectors.

TF-IDF Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to reflect the importance of a word in a document relative to a collection of documents (corpus). The implementation involves the following steps:

Term Frequency (TF): For each word in a review, the number of times it appears is counted. A term frequency vector is generated based on the vocabulary created earlier.

TF-IDF Calculation: The TF vector is multiplied element-wise with the IDF vector to generate the final feature representation for each review.

This TF-IDF vector acts as the numerical input for the classification model.

Logistic Regression (Manual Implementation): Instead of training a logistic regression model, the task simulates its behavior using manually defined weights.

The weights are chosen based on domain intuition — for example, words like "love", "best", and "product" are assigned positive weights, while other words receive negative weights.

A bias term is also included.

The logistic regression prediction is calculated using the sigmoid activation function, which converts the weighted sum of input features

into a probability between 0 and 1. The final sentiment classification is determined by applying a threshold: if the probability is greater than 0.5,

the review is considered positive; otherwise, it is negative.

Evaluation and Output: The model is tested on several new reviews. For each test review, the code prints:

1.The original review text.

2.The computed TF-IDF vector.

3.The predicted sentiment (Positive or Negative).

4.The probability score output by the sigmoid function.

This output provides both the raw feature representation and the model’s interpretation of the sentiment, helping to understand how different words influence the final prediction.

This task is highly valuable from a learning perspective, as it breaks down the core concepts of machine learning into simple, understandable steps.

Rather than using pre-built models and abstracted functions, this implementation focuses on how data is processed and classified under the hood.

By manually creating the TF-IDF vectors and simulating a logistic regression classifier, learners gain insight into:

1.Feature extraction from text

2.Importance of words using TF-IDF

3.Linear classification using weights and bias

4.Binary classification decision-making through sigmoid activation

**OUTPUT:**

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/c19dfb1a-1ce4-4ae2-b397-c3e6a1e3b8b5" />
