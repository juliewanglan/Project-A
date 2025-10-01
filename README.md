# Binary Sentiment Classifier
**Objective:** Build a machine learning model that classifies reviews into positive or negative sentiment categories. 

**Dataset:**
The dataset contains several thousand single-sentence reviews collected from three domains: imdb.com, amazon.com, yelp.com. Each review consists of a sentence and a binary label indicating the emotional sentiment of the sentence (1 for reviews expressing positive feelings; 0 for reviews expressing negative feelings). 

All the provided reviews in the training and test set were scraped from websites whose assumed audience is primarily English speakers, but of course may contain slang, misspellings, some foreign characters, and many other properties that make working with natural language data challenging (and fun!).

**Performance Metric:** Area Under the Operator Receiver Curve (AUROC) 

**Methods Used:**
1. Bag-of-Words Feature Representation with Logistic Regression classifier with L2 penalty
2. BERT embeddings with MLP classifier


**Results:**
Our AUROC for the test set per the leaderboard score was 0.96388. This was lower than our
training AUROC score of 0.9833, but comparable to our validation AUROC score of 0.9676. As our test
set score and heldout data scores are similar, it indicates our model is not overfit and can perform well
on unseen data. This is a significant improvement as compared to our performance on problem 1,
where we received an AUROC score of 0.863. This improvement is likely due to the BERT
embeddings, which is a richer and more informative feature vector representation. It’s ability to better
understand the meaning of words and phrases improved our model. This more complex representation
of our sentences, coupled with a classifier (MLP) that can recognize more intricate patterns as
compared to Logistic Regression likely resulted in better performance.

[Link to Project Description](https://www.cs.tufts.edu/comp/135/2024f/Assignments/projectA.html)
