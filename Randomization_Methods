# Randomization Methods

## Chi-squared test
**check if the counts of a categorical variable match expected probabilities**

**regular chi-squuared test is used when the data are a sample taken from the population**

Under H0 : P(C1&R1)=P(C1)*P(R1); independent

```
chisq.test(table)$expected #expected table
chisq.test(table) #test result
```

## McNemar's test
**similar to the chi-squared test, but used only for matched pairs**

```
mcnemar.test(table)
```

## Randomization test (resampling)
**non-random samples is common in experimentation**
Under H0 : the result is from pure randomization
```
yawning <- c( rep(T, 10), rep(F, 24), rep(T, 4), rep(F, 12)) #the result
treatment <- yawning[1:34]
control <- [35:50]
mean(treatment)
mean(control)
obs_dif<-mean(treatment)-mean(control)
```

```
#one time randomization
set.seed(1)
randomized <- sample(yawning)
groupA <- randomized[1:34]
groupB <- randomized[35:50]
mean(groupA)
mean(groupB)
rand_dif<-mean(groupA)-mean(groupB)
```

```
#many times randomization
set.seed(1)
differences <- rep(NA, 5000) #empty vector

for(i in 1:5000){
randomized <- sample(yawning)
randomized <- sample(yawning)
groupA <- randomized[1:34]
groupB <- randomized[35:50]
differences[i] <- mean(groupA)-mean(groupB)
}

summary(differences)
```

```
#empirical p-value
mean(abs(differences) >= obs_dif) #two sided test
mean(differences >= obs_dif) #one sided test
```

## Fisher's Exact Test
**similar to the chi-squared test, but uses permutations(순열) of the data to get an exact p-value**
**the count any one cell is below 5 & sample size below 50**

```
tr <- c(rep('treat', 34), rep('control', 16))
table <- table(yawning, tr)
fisher.test(table, alt="greater")
```

## Randomization testing for Numeric data
**for comparing means, we still test H0 : mu1=mu2, no longer assuming that our sample is randomly drawn from a population**
**Instead, we assume that every possible permutation of the random assignment of treatments is equally likely**

```
set.seed(123)
obs_dif <- mean(y[feet]) - mean(y[meter])

meandiffs <- double(10000) #double : floating point numbers
for (i in 1:length(meandiffs)) {
sy <- sample(y)
group1 <- sy[1:44]
group2 <- sy[45:113]
meandiffs[i] <- mean(group1) - mean(group2)
}

hist(meandiffs)
abline(v=obs_dif, lty=2)
abline(v=-obs_dif, lty=2)

greater <- abs(meandiffs) >= abs(obs_diff)
mean(greater) #empirical p-value
```

## Exact Independence test (package "coin")
**same senarios where we would use the randomization test(non-random sample)**
**difference : draw p-value by finding all the possible permutations of the data**

```
library(coin)
independence_test(y ~ unit, data=roomwidth) #asymptotic(점근선의)
independence_test(y ~ unit, data=roomwidth, distribution="exact") #exact #may be computationally expensive depending the data size
independence_test(y ~ unit, data=roomwidth, distribution="approximate") #approximate
```

## Exact wilcoxon test (non-parametric alternative to the Randomization test); not used in real life examples
**for random sample : wilcox.test**
```
wilcox_test(y ~ unit, data=roomwidth) #asymptotic(점근선의)
wilcox_test(y ~ unit, data=roomwidth, distribution="exact")
```

## Bootstrap (non parametric bootstrap, package "boot")
**data obtained via random sampling from the population**
**pretend your sample is the population**
**duplicate the data(with replacement)**
```
#alpha.fn returns the proportion to invest in X that will minimize the total variance of the quantity alpha*X + (1-alpha)*Y

alpha.fn <- function(data, index){
X <- data$X[index]
Y <- data$Y[index]
return( (Var(Y) - COV(X, Y)) / Var(X) + Var(Y) - 2*cov(X, Y)) )
}
alpha.fn(Portfolio, 1:100)

set.seed(1)
alpha.fn(Portfolio, sample(1:100, 100, replace = True))
boot(data=Portfolio, statistic=alpha.fn, R=1000)
```

## Cross-Validation
**employ resampling to test the validity of a proposed model**
```
set.seed(1)

train <- sample(392, 196)
lm.fit <- lm(mpg ~ horsepower, data=Auto, subset=train)
test_predict <- predict(lm.fit, Auto) [-train]
test_actual <- Auto$mpg [-train]
mean( (test_actual - test_predict)^2 )

lm.fit2 <- lm(mpg ~ poly(horsepower, 2), data=Auto, subset=train)
test_predict2 <- predict(lm.fit2, Auto) [-train]
mean( (test_actual - test_predict2)^2 )

lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data=Auto, subset=train)
test_predict3 <- predict(lm.fit3, Auto) [-train]
mean( (test_actual - test_predict3)^2 )

set.seed(2)

train <- sample(392, 196)
lm.fit <- lm(mpg ~ horsepower, data=Auto, subset=train)
test_predict <- predict(lm.fit, Auto) [-train]
test_actual <- Auto$mpg [-train]
mean( (test_actual - test_predict)^2 )

lm.fit2 <- lm(mpg ~ poly(horsepower, 2), data=Auto, subset=train)
test_predict2 <- predict(lm.fit2, Auto) [-train]
mean( (test_actual - test_predict2)^2 )

lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data=Auto, subset=train)
test_predict3 <- predict(lm.fit3, Auto) [-train]
mean( (test_actual - test_predict3)^2 )
```

## Leave-out-one Cross Validation (LOOCV) (package "boot")
```
library(boot)

cv.error <- rep(0, 5)
for (i in 1:5) {
glm.fit <- glm(mpg ~ poly(horsepower, i), data=Auto)
cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1]
}
```

##### mannual implementation
```
dim(Auto)
errors <- matrix(0, nrow=392, ncol=5)
for (k in 1:392) {
test <- Auto[k,]
train <- Auto[-k,]
for (i in 1:5) {
lm.fit <- lm(mpg ~ poly(housepower, i), data=train)
errors[k, i] <- (test$mpg - predict(lm.fit, test))^2
}
}

colMeans(errors)
```

## k-fold Cross-Validation (package "boot")
```
set.seed(17)
cv.error.10 <- rep(0, 10)
for(i in 1:10) {
glm.fit <- glm(mpg ~ poly(horsepower, i), data=Auto)
cv.error.10[i] <- cv.glm(Auto, glm.fit, K=10)$delta[1]
}
cv.error.10
```

##### mannual implementation
```
set.seed(17)
samp <- sample(392)
errors <- matrix(0, nrow=10, ncol=10)
for (k in 1:10) {
from <- 1 + (k-1)*39
to <- 39*k
test <- Auto[samp[from:to], ]
train <- Auto[samp[-(from:to)], ]
for (i in 1:10) {
lm.fit <- lm(mpg ~ poly(horsepower, i), data=train)
errors[k, i] <- mean( (test$mpg - predict(lm.fit, test))^2 )
}
}
colMeans(errors)
```

## Bootstrap applied to linear models

```
boot.fn <- function(data, index){
return(coef(lm(mpg ~ horsepower, data=data, subset=index)))
}
boot.fn(Auto, 1:392)

set.seed(1)
boot.fn(Auto, sample(392, 392, replace = T))

boot(Auto, boot.fn, 1000)
```
