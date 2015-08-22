# Predicting Human Activity Using Machine Learning in R
Daniel Ang  
August 22, 2015  

# Loading and Pre-Processing the Data
First, the data was downloaded from the specified website, and read into two variables (the training and the test sets). We also load the useful `caret` and `dplyr` packages:

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-training.csv",method = "curl")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-testing.csv",method = "curl")

df <- read.csv("pml-training.csv")
tf <- read.csv("pml-testing.csv")

library(caret)
library(dplyr)
```

The training data consists of 19,622 observables of 160 variables. We want to trim down the data such that we can cut down on the processing time while still making an effective model to predict on the test data. We first examine the test data, and discover that there are several columns which only consist of NAs. Taking these columns on the training set would be useless when we finally apply it to the training set, so we get rid of these columns in the training set:

```r
gdf <- df[,colSums(is.na(tf)) < nrow(tf)] 
```

We still observe that there are columns that consists of many zeros in the training set. The zeros in such columns are most likely not observations, but malfunctioning sensors or something similar. So it makes sense to eliminate them. In any case, they will not influence the models in a significant way even if there is still some data in them. We set a threshold of how many zeros in a column is too much. After playing around a little bit (including running our eventual model on different threshold levels), we decided to set the threshold at 2.5%:

```r
#now to get rid of columns with too many zeros:
coldata <- colSums(gdf ==0)

maxprop <- 0.025 #what is the max proportion of zeroes in a column we can allow?
rgdf <- gdf[coldata < maxprop * nrow(gdf)]
```

To prepare the data further for pre-processing and model fitting, we convert all the integer columns into numeric type.

```r
rgdf[,7:ncol(rgdf)-1] = data.frame(sapply(rgdf[,7:ncol(rgdf)-1],as.numeric)) # convert to numeric to allow PCA
```

Our data is now almost ready. To make processing easier, we eliminate the columns which contain the name, timestamp, and `new_window` (a Boolean variable which is unclear) - we only want the numbers from the various sensors.


```r
trgdf <- rgdf[,7:ncol(rgdf)] #subset out the columns to be used for training (classe and the numeric columns)
```

# Choosing a Model and Applying It
## Creating a Function for Convenient Parameter Searching
In the process of choosing this final model, we wanted to find a way to be able to easily switch between different types of models and their parameters. We created a custom function to do this. Let's go through the specifics of the function:

First, we want to try our models on a small sample of the training data (defined by the variable `n`), so we use the `sample_n` function without replacement:

```r
modClasse <- function(trgdf,n, partition = 0.75, pcamethod = "pca",tcmethod = "repeatedcv", tcnumber, tcrepeats, tcthresh = 0.95,fitmethod = "rf",allowParallel = FALSE,tuneGrid = NULL){
  set.seed(123)
  tsgdf <- sample_n(trgdf,n)
```
Then within this sample, we divide the data into training and test sets, because we want a subset of the data which is not used for cross validation or any other sort of model-fitting: a mock test set. This helps to prevent over-fitting:

```r
  inTrain <- createDataPartition(tsgdf$classe,p=partition,list = FALSE)
  training <- tsgdf[inTrain,]
  testing <- tsgdf[-inTrain,]
```
Next, we define a `trainControl` object that will pass on parameters into the `train` function. This mostly control the sampling, pre-processing, and cross validation procedures. Only parameters which we will play with are passed into it. In our case, `method` is important (we played around with bootstrapping vs. repeated cross validation), as well as `number` and `repeats` (controlling the number of folds in k-fold cross validation or the number of bootstrapping iterations, both of which are proportional to the overall processing time). We also played around with parallel processing, but ultimately found that the using multiple cores on the moderate-level laptop computer we had led to worse, not better performance.

```r
  fitControl <- trainControl(method = tcmethod,number = tcnumber,repeats = tcrepeats,preProcOptions = list(thresh = tcthresh),allowParallel = allowParallel)
```
The function is ready to fit the model itself, using the `train` function of the `caret` package. The `method` argument specifies which method to use (we experiment mostly with different types of random forests, and generalized linear models). The `tuneGrid` variable allows for finer control on the different parameters of the model (for example, the `mtry` variable for the `rf` method) - we tried this several times, often discovering that the automatic search done by the computer is already effective.

```r
  sampleFit <- train(classe ~ ., data=training,preProcess = pcamethod,method = fitmethod,trControl = fitControl,tuneGrid = tuneGrid)
```
The found model is then used to predict the outcomes (`classe`) of the test data (which we had earlier set aside from our initial `n` samples). The number of matching predictions is divided by the total number of test observables, resulting in a success rate which is returned by the function. We then close the function.

```r
sampleFit <- train(classe ~ ., data=training,preProcess = pcamethod,method = fitmethod,trControl = fitControl,tuneGrid = tuneGrid)
  print(sampleFit)
  
  pred <- predict(sampleFit,testing)
  predRight <- pred == testing$classe
  successrate <- length(predRight[predRight == TRUE])/length(pred)
  print(successrate)
  return(successrate)
}
```



## Deciding on the Model to Choose
We played around with the `modClasse` function defined above, typically setting `n`=2000, about a tenth of the actual training set. We paid attention to both the success rate as well as how the execution time scales (which is not always linear), because we will eventually have to apply the model to all 19,622 observables. In our model search, we noticed that the success rate generally went up as we increased `n`, even though `partition`, our variable that controlled the partition of the training and test sets, remained the same. We found that bootstrapping and repeated k-fold cross validation gave similar results, but the former usually took much longer, so we settled on the latter. In Repeated CV, we found that increasing the number of k-folds beyond 3 multiplied the execution time without significant increase in the success rate. The same goes for increasing the number of repetitions, which we kept at 1. 

We also tried preprocessing with principal components analysis, setting the `thresh` variable (controlling the number of PCs to be able to explain a percentage of the variance in the data) to typically 0.95. But this only slowed down and worsened the model, so we ultimately decided not to use PCA.

For the actual model itself, we did not find much variation between `glm` and `rf`, being able to achieve above 90% success rate with both. We tried different methods of random forests, including random ferns and oblique random forests, but these were not satisfactory, resulting in worse success rates or simply taking way too long. So we kept to the usual `rf` method. In evaluating different choices of models and model parameters, we often ran them several times (with different random seeds each time) and considered the average success rate.

We didn't have time to explore different models in great depth, however, we found that with `n`=6000 we were able to constantly achieve about 97% success rate:

```r
successvec = rep(0,5)

for (i in 1:5){
  successvec[i] = modClasse(trgdf = trgdf,n = 6000,tcnumber = 3, tcrepeats = 1,pcamethod = NULL,fitmethod = "rf")
}
```

```
## [1] 0.9766511
## [1] 0.9806409
## [1] 0.9792919
## [1] 0.9866489
## [1] 0.9752839
```

```r
mean(successvec)
```

```
## [1] 0.9797033
```
From our cross-validation techniques executed several times as seen above, we estimated the error rate to be 100 minus the success rate, which gives us 2%. When we are using the full training set instead of just 6000, presumably this error rate will be even lower. we were hence ready to apply the chosen model to the entire dataset.

## Applying the Model to the Entire Set
The code for doing this is similar to the code defining the `modClasse` function, except that we don't sample a small subset of the training set, instead simply setting

```r
tsgdf = trgdf
```
We also don't bother to further divide the training set into test and training subsets, instead applying the model algorithm directly to `tsgdf`. But before we do this, we first set the various model parameters that we have chosen:

```r
tcmethod = "repeatedcv"
tcnumber = 3
tcrepeats = 1
fitmethod = "rf"
allowParallel = FALSE

fitControl <- trainControl(method = tcmethod,number = tcnumber,repeats = tcrepeats,allowParallel = allowParallel)
```
Then we are ready to apply the model:

```r
fullFit <- train(classe ~ ., data=tsgdf,method = fitmethod,trControl = fitControl)
print(fullFit)
```

```
## Random Forest 
## 
## 19622 samples
##    36 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold, repeated 1 times) 
## Summary of sample sizes: 13082, 13081, 13081 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD   
##    2    0.9927633  0.9908452  0.0008823668  0.001116122
##   19    0.9965854  0.9956809  0.0014209554  0.001797275
##   36    0.9946488  0.9932309  0.0016112713  0.002038467
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 19.
```
## Predicting on the Actual Test Set
Then we finally apply `fullFit` to `tf`, the actual test set consisting of 20 observables:

```r
answers <- predict(fullFit,tf)
```
And we generate the text files to be submitted:

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```

Our model building and predicting is finally finished.
