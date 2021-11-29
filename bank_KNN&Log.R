library(ggplot2)
library(class)
library(lattice)
library(caTools)
library(caret)
library(dplyr)

bank.raw <- read.csv("bank.csv",sep = ";")

#Normalisation
Normalisation <- preProcess(bank.raw[,c(1,6,10,12:15)], method = c("center", "scale"))
bank.norm <- predict(Normalisation, bank.raw)
#Converting to numerial factors
bank.norm$default <- ifelse(bank.norm$default == "yes", 1,0)
bank.norm$housing <- ifelse(bank.norm$housing == "yes", 1,0)
bank.norm$loan <- ifelse(bank.norm$loan == "yes", 1,0)

bank.df <- bank.norm[,c(1,5:8,12:15)]

#job
job_admin <- ifelse(bank.norm$job == "admin.", 1,0)
job_unknown <- ifelse(bank.norm$job == "unknown", 1,0)
job_unemployed <- ifelse(bank.norm$job == "unemployed", 1,0)
job_management <- ifelse(bank.norm$job == "management", 1,0)
job_housemaid <- ifelse(bank.norm$job == "housemaid", 1,0)
job_entrepreneur <- ifelse(bank.norm$job == "entrepreneur", 1,0)
job_student <- ifelse(bank.norm$job == "student", 1,0)
job_bluecollar <- ifelse(bank.norm$job == "blue-collar", 1,0)
job_selfemployed <- ifelse(bank.norm$job == "self-employed", 1,0)
job_retired <- ifelse(bank.norm$job == "retired", 1,0)
job_technician <- ifelse(bank.norm$job == "technician", 1,0)
job_services <- ifelse(bank.norm$job == "services", 1,0)
bank.df <- cbind(bank.df, job_admin, job_unknown, job_unemployed, job_management, job_housemaid, 
                 job_entrepreneur, job_student, job_bluecollar, job_selfemployed,
                 job_retired, job_technician, job_services)
#marital status
marital_divorced <- ifelse(bank.norm$marital == "divorced", 1,0)
marital_married <- ifelse(bank.norm$marital == "married", 1,0)
marital_single <- ifelse(bank.norm$marital == "single", 1,0)
bank.df <- cbind(bank.df, marital_divorced, marital_married, marital_single)
#contact
contact_cellular <- ifelse(bank.norm$contact == "cellular", 1,0)
contact_telephone <- ifelse(bank.norm$contact == "telephone", 1,0)
contact_unknown <- ifelse(bank.norm$contact == "telephone", 1,0)
bank.df <- cbind(bank.df,contact_cellular, contact_telephone, contact_unknown)
#education
education_primary <- ifelse(bank.norm$education == "primary", 1,0)
education_secondary <- ifelse(bank.norm$education == "secondary", 1,0)
education_tertiary <- ifelse(bank.norm$education == "tertiary", 1,0)
bank.df <- cbind(bank.df ,education_primary, education_secondary, education_tertiary)
#poutcome
poutcome_failure <- ifelse(bank.norm$poutcome == "failure", 1,0)
poutcome_other <- ifelse(bank.norm$poutcome == "other", 1,0)
poutcome_success <- ifelse(bank.norm$poutcome == "success", 1,0)
bank.df <- cbind(bank.df,poutcome_failure, poutcome_other, poutcome_success)

bank.df$y <- ifelse(bank.norm$y == "yes", 1,0)

#create training and test set
set.seed(123)
train.index <- sample(row.names(bank.df), 0.8*dim(bank.df)[1])
valid.index <- setdiff(row.names(bank.df), train.index)
bank.train <- bank.df[train.index, ]
bank.valid <- bank.df[valid.index, ]
t(t(names(bank.valid)))

### USE KNN ###
#predicting
# knn.pred <- knn(train = bank.train,
#                 test = bank.valid,
#                 cl = bank.train$y, k = 3)
# knn.pred

#codes for measuring the accuracy of different k values
accuracy.df <- data.frame(k = seq(1, 15, 1), accuracy=rep(0, 15))
for(i in 1:15){
  knn.pred <- knn(train = bank.train,
                  test = bank.valid,
                  cl = bank.train$y, k = i)
  accuracy.df[i,2] <- confusionMatrix(knn.pred, 
                                      as.factor(bank.valid$y))$overall[1]
}
accuracy.df
which(accuracy.df[,2] == max(accuracy.df[,2])) 

#knn with k = 3
knn.pred.train <- knn(train = bank.train,
                      test = bank.train,
                      cl = bank.train$y, k = 3)
confusionMatrix(knn.pred.train,
                as.factor(bank.train$y), positive = "1")


knn.pred.valid <- knn(train = bank.train,
                      test = bank.valid,
                      cl = bank.train$y, k = 3)
confusionMatrix(knn.pred.valid,
                as.factor(bank.valid$y), positive = "1")

###Logistic Regression###
bank.reg <- glm(y ~ ., data = bank.df[,-c(21,24,27)], family = "binomial")
options(scipen=999)
summary(bank.reg)

heatmap(cor(bank.df), Rowv = NA, scale = "column")


log.pred.train <- predict(bank.reg, bank.train, type = "response")

ggplot(bank.train, aes(x = y)) +
  geom_density(lwd = 0.5) +
  labs(title = "Distribution of Probability Predication Training Data") +
  theme_minimal()

#cutoff = 0.5
train.class <- (1*(log.pred.train>0.5))

confusionMatrix(as.factor(train.class), as.factor(bank.train$y), positive = "1")


log.pred.valid <- predict(bank.reg, type = "response", newdata = bank.valid)

ggplot(bank.valid, aes(x = y)) +
  geom_density(lwd = 0.5) +
  labs(title = "Distribution of Probability Predication Validation Data") +
  theme_minimal()

#cutoff = 0.5
valid.class <- (1*(log.pred.valid>0.5))

confusionMatrix(as.factor(valid.class), as.factor(bank.valid$y), positive = "1")

