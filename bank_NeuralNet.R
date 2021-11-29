library(ggplot2)
library(class)
library(lattice)
library(caTools)
library(caret)
library(dplyr)
library(neuralnet)

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

y <- ifelse(bank.norm$y == "yes", 1,0)
bank.df <- cbind(bank.df,y)

#Splitting the training and validation sets
set.seed(123)
train.index <- sample(row.names(bank.df), 0.8*dim(bank.df)[1])
valid.index <- setdiff(row.names(bank.df), train.index)
bank.train <- bank.df[train.index, ]
bank.valid <- bank.df[valid.index, ]

#names(bank.train)


#Neural Net
#Single 5
model_single5 <- neuralnet(factor(bank.train$y) ~ age + default + balance + housing + loan + duration + campaign + pdays 
                           + previous + job_admin + job_unknown + job_unemployed + job_management + job_housemaid 
                           + job_entrepreneur + job_student + job_bluecollar + job_selfemployed + job_retired
                           + job_technician + job_services + marital_divorced + marital_married + marital_single     
                           + contact_cellular + contact_telephone + contact_unknown + education_primary
                           + education_secondary + education_tertiary + poutcome_failure + poutcome_other + poutcome_success,
                           data = bank.train, linear.output = F,
                           hidden = 5)

plot(model_single5)
model_single5$weight

#predictions on training and validation data
options(scipen = 0)
#training predictions
train.pred.single5 <- compute(model_single5, bank.train)
train.pred.single5 <- train.pred.single5$net.result[,2]
# convert probabilities to classes
train.class.single5 <- (1* (train.pred.single5 > 0.5)) 
confusionMatrix(factor(train.class.single5), factor(bank.train$y), positive = "1")

valid.pred.single5 <- compute(model_single5, bank.valid)
valid.pred.single5 <- valid.pred.single5$net.result[,2]
# convert probabilities to classes
valid.class.single5 <- (1* (valid.pred.single5>0.5))
# confusion matrix 
confusionMatrix(factor(valid.class.single5), factor(bank.valid$y), positive = "1")


#Single 12
model_single12 <- neuralnet(factor(bank.train$y) ~ age + default + balance + housing + loan + duration + campaign + pdays 
                            + previous + job_admin + job_unknown + job_unemployed + job_management + job_housemaid 
                            + job_entrepreneur + job_student + job_bluecollar + job_selfemployed + job_retired
                            + job_technician + job_services + marital_divorced + marital_married + marital_single     
                            + contact_cellular + contact_telephone + contact_unknown + education_primary
                            + education_secondary + education_tertiary + poutcome_failure + poutcome_other + poutcome_success,
                            data = bank.train, linear.output = F,
                            hidden = 12)

plot(model_single12)
model_single12$weights

#predictions on training and validation data
options(scipen = 0)
#training predictions
train.pred.single12 <- compute(model_single12, bank.train)
train.pred.single12 <- train.pred.single12$net.result[,2]
# convert probabilities to classes
train.class.single12 <- (1* (train.pred.single12 > 0.5)) 
confusionMatrix(factor(train.class.single12), factor(bank.train$y), positive = "1")

valid.pred.single12 <- compute(model_single12, bank.valid)
valid.pred.single12 <- valid.pred.single12$net.result[,2]
# convert probabilities to classes
valid.class.single12 <- (1* (valid.pred.single12>0.5))
# confusion matrix 
confusionMatrix(factor(valid.class.single12), factor(bank.valid$y), positive = "1")


#Two 2
model_two2 <- neuralnet(factor(bank.train$y) ~ age + default + balance + housing + loan + duration + campaign + pdays 
                            + previous + job_admin + job_unknown + job_unemployed + job_management + job_housemaid 
                            + job_entrepreneur + job_student + job_bluecollar + job_selfemployed + job_retired
                            + job_technician + job_services + marital_divorced + marital_married + marital_single     
                            + contact_cellular + contact_telephone + contact_unknown + education_primary
                            + education_secondary + education_tertiary + poutcome_failure + poutcome_other + poutcome_success,
                            data = bank.train, linear.output = F,
                            hidden = c(2,2))

# When the neuralnet does not converge, the resulting neural network will be incomplete. 
# You can determine this by calling attributes(fit)$names
# attributes(model_two2)$names

plot(model_two2)
model_two2$weights

#predictions on training and validation data
options(scipen = 0)
#training predictions
train.pred.two2 <- compute(model_two2, bank.train)
train.pred.two2 <- train.pred.two2$net.result[,2]
# convert probabilities to classes
train.class.two2 <- (1* (train.pred.two2 > 0.5)) 
confusionMatrix(factor(train.class.two2), factor(bank.train$y), positive = "1")

valid.pred.two2 <- compute(model_two2, bank.valid)
valid.pred.two2 <- valid.pred.two2$net.result[,2]
# convert probabilities to classes
valid.class.two2 <- (1* (valid.pred.two2>0.5))
# confusion matrix 
confusionMatrix(factor(valid.class.two2), factor(bank.valid$y), positive = "1")

library(gains)
train.gain <- gains(as.numeric(bank.valid$y),valid.pred.single5, groups = 10)
barplot(train.gain$mean.resp/mean(bank.train$y),
        names.arg = train.gain$depth,
        xlab = "Depth of Record", ylab = "Mean Response",
        main = "NeuralNet Decile-wise lift chart")

