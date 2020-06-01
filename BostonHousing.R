install.packages('mlbench')
install.packages('corrplot')
install.packages('Amelia')
library.packages('MASS')
library.packages('caret')
install.packages('caTools')
install.packages('lmtest')
install.packages('olsrr')
install.packages('nortest')
install.packages('tsoutliers')
install.packages('mctest')
install.packages('car')

library(mlbench)
library(corrplot)
library(Amelia)
library(MASS)
library(caret)
library(caTools)
library(lmtest)
library(olsrr)
library(nortest)
library(tsoutliers)
library(mctest)
library(car)

#load the data

data(BostonHousing)
boston<-BostonHousing
head(boston)
str(boston)

#chas, rad are binary and ordinal data types

unique(boston$rad)
unique(boston$chas)

#view data
boston
View(boston)

#summary statistics

summary(boston)

# Rename a column in R
#colnames(boston)[colnames(boston)=="medv"] <- "med_val"

#subset the data - exclude chas and rad

var_col<-c("crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "b", "lstat", "medv")
boston<-boston[var_col]

View(boston)

#check missing data - location

which(is.na(boston))
colSums(is.na(boston))

#map the missing data cases

missmap(boston, col=c('red', 'black'), y.at=1, y.label='', legend=TRUE)
missmap(boston,  y.at=1, y.label='', legend=TRUE)

#correlation matrix

cor(boston, method="pearson")

?cor

#corrplot(cor(boston), method="number")
# Specialized the insignificant value according to the significant level

# matrix of the p-value of the correlation

p.boston <- cor.mtest(boston)$p
p.boston


## add all p-values
corrplot(cor(boston), p.mat = p.boston , insig = "p-value", sig.level = -1)

corrplot(cor(boston), type = "upper", order = "hclust", 
         p.mat = p.boston, sig.level = 0.01)


#correlation plot of all variables

pairs(boston)

##use corr plot
corrplot(cor(boston), method="circle")
corrplot(cor(boston), method="ellipse")

#correlation plot of selected cols
pairs(~ medv + indus + rm +  tax + ptratio+ lstat , data=boston, main = "Boston Housing Data")

#https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html


#correlation sig test

cor.test(boston$crim, boston$medv, method="pearson")

cor.test(boston$lstat, boston$medv, method="pearson")

#lstat -0.74//rm=0.69 // ptratio=-0.51//indus=-0.48//tax=-0.47//crim=-0.388//zn=0.36//b=0.333//nox=-0.42

#check
#univariate model

bmodel<-lm(medv~lstat, data=boston)
summary(bmodel)

#multiple regression

bmodel<-lm(medv~., data=boston)
summary(bmodel)

##stepwise regression

stepbmodel<-stepAIC(bmodel, direction="backward")
summary(stepbmodel)
stepbmodel$anova

##build based on best model from stepwise

bmodelfinal<-lm(medv ~ crim + zn + nox + rm + dis + ptratio + b + lstat, data=boston)
summary(bmodelfinal)

par(mfrow=c(1,1)) 

plot(bmodelfinal)

##build based on best model from stepwise
bmodelfinal<-lm(medv ~ nox + rm + dis + ptratio + b + lstat, data=boston)
summary(bmodelfinal) 
par(mfrow=c(1,1)) 
plot(bmodelfinal)
par(mfrow=c(2,2)) 


#Breusch Pagan Test for heteroskedasticity

bptest(bmodelfinal)

#error constant test
ols_plot_resid_fit(bmodelfinal)

?ols_plot_resid_fit()

#multicollinearity test 
#Farrar Glauber Test
indvar<-c("nox", "rm",  "dis",  "ptratio", "b", "lstat")
X<-boston[indvar]
dvar<-c("medv")
Y<-boston[dvar]
omcdiag(X,Y)
imcdiag(X,Y)


#vif
#https://cran.r-project.org/web/packages/olsrr/vignettes/regression_diagnostics.html
#The general rule of thumb is that VIFs exceeding 4 warrant further investigation, while VIFs exceeding 10 are signs of serious multicollinearity requiring correction.

ols_vif_tol(bmodelfinal)
?ols_vif_tol()

vif(bmodelfinal)

?vif()
#exists!
#https://www.r-bloggers.com/multicollinearity-in-r/

#You are correct that the null hypothesis of the Breusch-Pagan test is homoscedasticity (= variance does not depend on auxiliary regressors). 
#If the p value less than 0.05, we reject null hypothesis�-value becomes "small", the null hypothesis is rejected.
#we reject errors is homoscedasticity 

#subset
indvar<-c("crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "b", "lstat")
X<-boston[indvar]

dvar<-c("medv")
Y<-boston[dvar]

#rfe
control <- rfeControl(functions=lmFuncs, method="repeatedcv", repeats=5)
# run the RFE algorithm
results <- rfe(X, Y, rfeControl=control)
# summarize the results
print(results)

###TESTING

View(boston)
#Split the data , `split()` assigns a booleans to a new column based on the SplitRatio specified. 
set.seed(101)
split <- sample.split(boston,SplitRatio =0.80)

train <- subset(boston,split==TRUE)
test <- subset(boston,split==FALSE)

View(train)
View(test)

#boston


indexdt = sample(1:nrow(boston), size=0.8*nrow(boston))

train <- boston[indexdt,]
test <- boston[-indexdt,]

model <- lm(medv ~ crim + zn+ nox + rm + dis + ptratio + b + lstat, data = train)
summary(model)

#View(boston)

predicted=predict(model, test)
result_preds<-data.frame(cbind(test, predicted))
result_preds


# train <- select(train,-b)
# test <- select(test,-b)
# Test for Autocorrelated Errors

durbinWatsonTest(bmodelfinal)

crPlots(bmodelfinal)
##normality test for residuals

ols_test_normality(bmodelfinal)

#Ad test
?ad.test()
ad.test(bmodelfinal$residuals)
shapiro.test(bmodelfinal$residuals)
?shapiro.test()
#JB test

JarqueBera.test(bmodelfinal$residuals)

#less than 0.05, reject null, not from normal 
#If the data comes from a normal distribution, the JB statistic asymptotically has a chi-squared distribution with two degrees of freedom, so the statistic can be used to test the hypothesis that the data are from a normal distribution. The null hypothesis is a joint hypothesis of the skewness being zero and the excess kurtosis being zero. '


