import math

# Step 1: Define a small dataset
documents = [
    ("I love this product", 1),
    ("This is the best thing ever", 1),
    ("I hate this", 0),
    ("This is awful and terrible", 0),
]

# Step 2: Tokenize and build vocabulary
def tokenize(text):
    return text.lower().split()

vocab = set()
for doc, _ in documents:
    vocab.update(tokenize(doc))

vocab = sorted(list(vocab))
word_to_index = {word: i for i, word in enumerate(vocab)}

# Step 3: Compute IDF
def compute_idf():
    N = len(documents)
    df = [0] * len(vocab)
    for i, word in enumerate(vocab):
        for doc, _ in documents:
            if word in tokenize(doc):
                df[i] += 1
    idf = [math.log((N + 1) / (df[i] + 1)) + 1 for i in range(len(vocab))]  # Smoothed IDF
    return idf

idf = compute_idf()

# Step 4: Compute TF
def compute_tf(doc):
    words = tokenize(doc)
    tf = [0] * len(vocab)
    for word in words:
        if word in word_to_index:
            tf[word_to_index[word]] += 1
    return tf

# Step 5: Compute TF-IDF
def compute_tfidf(doc):
    tf = compute_tf(doc)
    tfidf = [tf[i] * idf[i] for i in range(len(vocab))]
    return tfidf

# Step 6: Simulated Logistic Regression (with hardcoded weights)
weights = [0.6 if word in ['love', 'best', 'product'] else -0.6 for word in vocab]
bias = -0.2

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_sentiment(doc):
    tfidf = compute_tfidf(doc)
    weighted_sum = sum(tfidf[i] * weights[i] for i in range(len(vocab))) + bias
    probability = sigmoid(weighted_sum)
    prediction = 1 if probability > 0.5 else 0
    return prediction, probability, tfidf

# Step 7: Test Sentences
test_reviews = [
    "I love this",
    "This is terrible",
    "Best purchase ever",
    "I hate this product",
    "Awful and useless",
    "I really like this thing"
]

print("=== Sentiment Analysis Results ===\n")
for review in test_reviews:
    pred, prob, tfidf_vector = predict_sentiment(review)
    print(f"Review: {review}")
    print("TF-IDF Vector:", ["{:.2f}".format(x) for x in tfidf_vector])
    print("Predicted Sentiment:", "Positive" if pred == 1 else "Negative")
    print("Probability Score: {:.4f}".format(prob))
    print("-" * 50)
