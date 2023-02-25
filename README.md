# CreditRiskPrediction
A basic overview of how you can use stratified k-fold analysis to estimate the predictive power of different models.
The Code is heavily commented to explain each step.

I use a logistic regression in this case and measure the Positive Predictive Value of different thresholds to predict credit risk default

Based on the dataset from: https://www.projectpro.io/article/projects-on-machine-learning-applications-in-finance/510

Many possible improvements could be made to this code from which data is used to build the model, to the model selection method and the models used.

In an enterprise environment with more computing resources, we could use the rclust package to run a triple nested loop and select the best model
out of thousands.
