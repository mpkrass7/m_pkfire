---
title: "Nested Cross Validation is an Important Thing"
author: Marshall Krassenstein
date: '2022-07-07'
slug: nested-cross-validation
categories: []
tags: [python, partitioning, machine-learning]
subtitle: ''
summary: 'A tutorial on what nested cross validation is, why it is important and how to implement it in Python'
authors: [Marshall Krassenstein]
featured: yes
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


In this tutorial I will provide an explanation of what nested cross validation is, why you should do it, and how to implement in Scikit Learn. 

### Prerequisites

You should read on in this tutorial if:
- Explanation: You have a basic understanding of what cross validation is
- Implementation: You have at least passing familiarity with the `scikit-learn` package

## What we already know about partitioning

If your job title has the phrase 'Data Science' or 'Machine Learning' in it, you're probably familiar with the idea of partitioning your data before you fit a model. Whether you are fitting just one model with a predefined set of hyper-parameters or many models as well, you need some way to evaluate how good they are. Evaluating on the same data you trained on produces overfit results where the most wiggly models perform amazingly well. At scoring time, you discover that they're hot pieces of trash when you have to produce actual predictions. If you don't believe me on this point, well, this article probably isn't for you. So, normally we split our data into a **training set** to build the model and a test or **holdout set** to evaluate the model. Ideally, this holdout set tells us how good our model is at predicting on unseen data. So, we might split our data like this:

![train and holdout set](images/train_holdout.png)

Now if this article is for you, then you probably also realize that you need more than just a training and a holdout set if you want to evaluate multiple models or if you want to tune hyper-parameters. Ideally, we need one more dataset that will help us choose the best tool for the problem, which we can then pass to the holdout set to evaluate that tool. We call this 'choose your model and hyper-parameter space' set the **validation set**. And that might look like this:

![train validate and holout set](images/train_validate_holout.jpeg)

Now before we go on, please do pay attention to the above claim and make sure you understand why you can't use your holdout set both for choosing between hyper-parameter options **and** for evaluating the chosen model. I will return to this idea, because it is the most important piece to understanding why nested cross validation is so important. 

But first I want to talk through one more point. Now, continuing the hypothesis that you are the intended audience of this article, I will further surmise that you are also familiar with cross-validation, the magical and commonly used data slicing strategy that lets you turn your training data into a debiased validation set. Effectively, you take your training dataset cut it into *n* equally sized pieces, fit a model using all of the data in *n-1* pieces, then evalute on the remaining piece. You repeat this process until each piece has been used as the single evaluating dataset. Cross validation looks like this:

![cross validation](images/cross_validation.jpeg)


Using cross validation gives you two big advantages: First you don't have to sacrifice some fit you would have achieved given more data to an entirely separate validation set. All of the data that isn't in your holdout set can be used to train the model. Second, it actually provides even more data to validate your model on, because you are also using the entire traing set to validate your model albeit indirectly. 

So, cross validation is great. The only real downside to cross validation is that doing it requires you to fit a model once for every **fold**, or the number of times you want to split your data. So, if you use five fold cross validations, you could expect about 5x more time needed to fit your model.


### So what's the Problem?

Up to this point, we've defined our train, validation, and holdout sets. We even talked about a nifty algorithm for merging the training and validation sets so that we can both fit our model and select its hyper-parameters using the same set of data. You might wonder, "Maybe there's some secret mathematical issue with Cross Validation?" But the issue is not with cross validation at all. In fact, if you truly do just use cross validation to train and select your model, leaving evaluation to your holdout set, then there actually isn't anything wrong with the above process. 

The problem is that most of the time we do not actually follow this process. To explain why most people don't do this and why that presents us with a problem, I will leave off cross validation for now and return to a regular train, validate, holdout partitioning scheme.

### Introducing Leakage

Let's say I want to train a Random Forest classifier on a dataset. I'm not sure how deep the trees should be, and I'm not sure how many trees to use. Being the classy practitioner I am, I decide to define a grid of hyper-parameters, maybe with tree depths of 3, 5 and 10 and a number of trees of 30, 100, and 200. That grid might look like this:

![trees](images/grid_search.jpg)

Since I have my validation set, I can train a model on each of these trees, and then find the best hyper-parameter space by choosing the best performing model. Maybe a tree depth and number of trees combination of 5 and 200 is the best performer. If I want to know how well it performed, I can assess my (5,200) random forest on whatever metrics interest me using the holdout dataset.  So far, so good. Except, wait, my performance on the holdout set really is not *that* much better than some of the other models. In fact, it seems like if I only trained a model with 100 trees, I could cut the time it takes to train in half with a minimal loss in performance. Maybe evaluating that model on the holdout set will even perform *better* when I look at some other metrics of evaluation than what I had originally considered. And actually, I didnt' test some other hyper-parameters. A model using 70 trees could perform just as well as one with 100 trees. Oh, modeling is such an *iterative process*.

So, I could restart the modeling process with a new hyper-parameter space. And then I can keep following this process until I have a model with satisfactory performance on the holdout set. But there's a problem here. If I keep redoing my models until my holdout score is adequate, I'm essentially just gaming the model fitting process because some configuration of the model is destined to perform better than others on *this sample of data*. Another phrase for this might be that I'm indirectly introducing **leakage** to my training data by incorporating my knowledge of performance on the holdout set to the modeling process. At the end of the day, I should not use the holdout set iteratively. It is meant to be evaluated at the latest stage of the modeling process.

### Validating on the validation set

Alright, so we know I should only use the holdout set to evaluate the final model, not to choose between models. What can I do instead? My next instinct might be to use the validation set as a preliminary performance evaluation for my model. After all, the model I choose to evaluate on the holdout set is the result of comparing a performance metric for each model trained using the validation set.  The problem with doing this is that one model is destined to perform better than every other model we fit even if it is not necessarily a better model. A thought experiment might illustrate this idea. 

Let's say I have 3 friends named Jonathan, Joseph, and Jolyne. I'm trying to decide who I should bring to a casino with me but I think they have different levels of luck. I hypothesize that when one of them flips a coin and the coin is supposed to be heads, it will be more likely to be heads and vice versa. I make a table of 10 coin flips I want and have each of them flip a coin ten times. The results might look like this:

![Coin Flips](images/coin_flips.jpg)

Amazing! Joseph flipped all but two of the coins exactly how he was supposed to. Good thing I'm not bringing Jonathan who must be very unlucky.

Obviously, this was a bit of a silly hypothesis. But what I hope it illustrates is that someone is destined to win even here even when there isn't a difference between three candidates. In the same vein, when you draw conclusions about how good your model is based on the exact dataset you used to select the model in the first place you are likely to get an overly optimistic evaluation of performance. 

### Enter Nested CV

Cross validation suffers from exactly the same problem as making a validation split. If it is used to decide between model hyper-parameters, it shouldn't also be used to evaluate the winning model. This is exactly what you can read in the [Scikit-Learn documentaiton](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), which also explains what Nested Cross validation is for:

*"..Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score... Nested cross-validation (CV) is often used to train a model in which hyper-parameters also need to be optimized. Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search."*

#### How Nested CV works

Scikit-Learn docs can also give us insight into how nested cross validation works: 

*"Nested CV effectively uses a series of train/validation/test set splits. In the inner loop, the score is approximately maximized by fitting a model to each training set, and then directly maximized in selecting (hyper)parameters over the validation set. In the outer loop, generalization error is estimated by averaging test set scores over several dataset splits."*

So basically, what's happening is a process like this:
- Separate out your holdout set. This will not be involved in the model fitting process.
- Use cross validation on your remaining dataset.
- For each iteration:
    - Consider the validation fold as your holdout set. This is our *outer cross validation* and its results will be used to evaluate the performance of the model obtained from our next step.
    - In the remaining dataset, perform cross validation again. This inner loop of cross validtaion will be used to determine a set of hyper-parameters.
    - When the optimal hyper-parameters are found, retrain your model on all of the data in this inner fold and evaluate this model on your outer cv validation set
- From this we end up with a debiased version of the same evaluation produced by Cross Validation. 

We can take our results from the above process and use them to compute whatever evaluation metrics we'd like with knowledge that they are unbiased despite us not actually creating a new dataset. We also have up to *i* different sets of hyper-parameters we can use for our final model. In most implementations our final model is the one trained using the first optimal set of hyper-parameters found.

The splits for nested cross validation are shown visually in the diagram below. If we have a 5 fold inner cv and a 5 fold outer cv, we will build 25 models * the number of hyper-parameter combinations we check in this process:

![nested_cv](images/illustrate_nested_cv.jpeg)


### Implementing Nested CV

Now from what I explained above, we could infer that running nested cv involves a nested `for` loop. The outer loop would run *i* times where *i* represents the number of folds in our outer cv. The inner loop would run *j* times where *j* represents the number of folds run on the inner cv. So the total number of models trained in this nested loop will be $i * j$. In practice we actually don't even need to write this unless we want more control over the output because the scikit-learn library has functions built in to help us. 

The scikit-learn library makes implememnting Nested CV very easy. Essentially, we just specify how many folds we want for our inner and outer cross validation paritioning schemes. Then, we pass the inner specification to `GridSearchCV` and the outer specification to `cross_val_score`. The result will be an estimator fitted with optimal hyper-parameters and a debiased cross validation score derived from evaluating our hyper-parameters on the outer folds of our dataset. 

To show this, I'll use the [wine quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) found inthe UCI Machine Leanring Repository. In my implementation below, I actually do use one loop, but the reason I'm doing this is to run nested cross validation in 10 experiments. This will let us more clearly see a difference between using nesting and not using nesting. The only issue is that we are essentially training 10 (rounds) * 4 (outer cv folds) * 5 (inner cv folds) * 6 (number of parameter combinations) = 1200 models!


```python
#Load libraries
from logzero import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split

import matplotlib.pyplot as plt
import warnings
%matplotlib inline
warnings.filterwarnings("ignore")

SEED = 42
ROUNDS = 10
TARGET = 'quality'

#Load data
data = pd.read_csv('data/winequality-red.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split data
X, y = data.drop(TARGET, axis=1), data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify = y)
```


```python
#Define the hyper-parameter grid
param_grid = {'max_depth': [10, 50],
                'n_estimators': [100, 200, 400]}

# Define our model
rf = RandomForestClassifier(random_state=SEED)

#Create arrays to store the scores
outer_scores = np.zeros(ROUNDS)
nested_scores = np.zeros(ROUNDS)


# Loop for each round
for i in range(ROUNDS):
    logger.info(f"Testing nested against non nested cross validation for model evaluation round {i + 1}...")
   
    # 5 fold cross validation...
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    # Run on each of the 4 outer folds
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
    
    # Run grid search to tune hyper-parameters (Note that this just runs on the inner cv)
    estimator = GridSearchCV(rf, param_grid=param_grid, cv=inner_cv, n_jobs=-1, scoring='neg_log_loss') 
    estimator.fit(X_train, y_train)
    
    # Append results of inner CV to outer score
    outer_scores[i] = estimator.best_score_

    # Now that we theoretically have our hyper-parameters set, we use the outer cv to actually score the model 
    nested_score = cross_val_score(estimator, X=X_train, y=y_train, cv=outer_cv, n_jobs=-1, scoring='neg_log_loss') 
    
    # Append results of round to nested scores
    nested_scores[i] = nested_score.mean()


```

    [I 220707 15:52:34 3194561931:15] Testing nested against non nested cross validation for model evaluation round 1...
    [I 220707 15:53:08 3194561931:15] Testing nested against non nested cross validation for model evaluation round 2...
    [I 220707 15:53:42 3194561931:15] Testing nested against non nested cross validation for model evaluation round 3...
    [I 220707 15:54:18 3194561931:15] Testing nested against non nested cross validation for model evaluation round 4...
    [I 220707 15:54:53 3194561931:15] Testing nested against non nested cross validation for model evaluation round 5...
    [I 220707 15:55:33 3194561931:15] Testing nested against non nested cross validation for model evaluation round 6...
    [I 220707 15:56:14 3194561931:15] Testing nested against non nested cross validation for model evaluation round 7...
    [I 220707 15:56:54 3194561931:15] Testing nested against non nested cross validation for model evaluation round 8...
    [I 220707 15:57:39 3194561931:15] Testing nested against non nested cross validation for model evaluation round 9...
    [I 220707 15:58:26 3194561931:15] Testing nested against non nested cross validation for model evaluation round 10...


Now let's look at the difference between our scores using nested and non nested cross validation. Below, I compute the average difference of scores and then plot the [log loss](https://www.kaggle.com/code/dansbecker/what-is-log-loss/notebook) values for each experiment. We can see that across the board, the log loss values are higher (i.e. worse) when we do nested CV than non nested CV. Does that mean we have a worse model using this method of cross validation? No! The results from Nested CV are simply a more accurate representation of how your model actually performs than the non nested CV scores.


```python
#Take the difference from the non-nested and nested scores
score_difference = outer_scores - nested_scores

print("Avg. difference of {:6f} with std. dev. of {:6f}."
      .format(score_difference.mean(), score_difference.std()))
```

    Avg. difference of 0.033996 with std. dev. of 0.016649.



```python
def plot_experiment(outer_scores, nested_scores, outpath=None):
    # Plot scores on each round for nested and non-nested cross-validation
    fig, ax = plt.subplots(figsize=(16,8))
    fig.tight_layout()
    outer_scores_line, = ax.plot(np.abs(outer_scores), linewidth = 4, color='orange')
    nested_line, = ax.plot(np.abs(nested_scores), linewidth=4, color='steelblue')

    ax.tick_params(axis="both", which="both", length=0, labelsize=20)
    ax.grid(False)

    ax.set_ylabel("Score", fontsize="20")
    ax.set_xlabel("Experiment Number", fontsize="20")

    plt.legend([outer_scores_line, nested_line],
              ["Non-Nested CV", "Nested CV"],
               prop={'size': 22})
    plt.title("Non-Nested vs Nested Cross-Validation on the Wine Dataset",
             x=.5, y=1, fontsize="24")
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')

plot_experiment(outer_scores, nested_scores, outpath="images/experiment_results.png")
```


    
![png](images/index_15_0.png)


### Concluding

Cross validation is standard practice in Machine Learning yet most organizations that do this create overly optimistic models. Among other reasons, this contributes to a frequently seen issue where models underperform in production compared to their performance in the building phase. On its own, nested Cross Validation does not provide us with a better model. And as we saw above, the computational cost of using Nested CV can be high, especially with many hyper-parameters and estimators. In exchange for time and compute power, it does allow us to properly and iteratively evaluate our models to produce realistic estimates of performance without using our holdout set. 


## References
- KDNuggets https://www.kdnuggets.com/2020/10/nested-cross-validation-python.html
- AnalyticsVidhya: https://www.analyticsvidhya.com/blog/2021/03/a-step-by-step-guide-to-nested-cross-validation/
- Arxiv: https://arxiv.org/abs/1809.09446

Credits to Kevin Arvai and Justin Swansberg for auditing my understanding of this topic
