# Create-model-and-test-with-injection-molding-data-automatically
Create model and test with injection molding data automatically

There's some cases to create model based on several companies's data saved in mongoDB

1. non labeld but have several X_features
using Clustering methods to create label(hard to measure of accuracy_score of model this is because we don't know what is the exact label of data)

2. labeled data with one to one matched
using Autoencoder methods for prediction label
using stacking and machine learning model for prediction label

3. labed data with batch to batch matched 
using Autoencoder methods for prediction label(in this cases we can't know the exact accuracy_score of model)
using clustering methods to create label(also can't measure exact accuracy_score of model)
