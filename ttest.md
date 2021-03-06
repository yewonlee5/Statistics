# Sampling distributions and t-test

## sampling distributions
```
set.seed(1)
popuation <- round(nnorm(10^6, mean=100, sd=16))
hist(population, breaks=30, xlim=c(40,160))

mean(population)
sd(population)

set.seed(1)
samp1 <- sample(population, size=10) # generate random samples

samples_means <- rep(NA, 10^4) # empty vector
for(i in 1:length(sample_means){
samp <- sample(population, size=10)
sample_means[i] <- mean(samp)
}
hist(sample_means, breaks=30, xlim=c(80,120))

mean(sample_means)
sd(sample_means)
```

## Central Limit Theorem
The sampling distribution of the sample mean x_bar will follow the normal distribution no matter what the parent population looks like as long as the samples are drawn independently and the sample is large enough.

## P-value
The probability of observing the data if we assume the null hypothesis is true.

## Student's two-sample t-test (parametric test) : depending on CLT, assuming the sampling distribution of the mean follows a normal distribution
**independent of each other**
**same variance**
```
t.test(estimate ~ unit, data=roomwidth, var.equal=T, alternative="two.sided") #var.equal default=F
```

## Welch two-sample t-test
**independent of each other**
**different variances**
```
t.test(estimate ~ unit, data=roomwidth) #fewer degrees of freedom, wider t-distribution, , larger p-value, less likely to reject null hypothesis
```

## Wilcoxon rank-sum test
*non-parametic test*
*no distributional assumptions about the sampling distribution*
*relatively small sample (less than 25)*
*generally less powerful than parametric counterparts(cost of fewer assumptions)*
```
wilcox.test(estimate ~ unit, data=roomwidth)
```

## Paired t-test
```
t.test(y1, y2, paired=T)
```

## Wilcoxon signed rank test
*use rank information, ties in the data matters*
*jittering(adding a small random positive or negative number)*
```
wilcox.test(y1, y2, paired=T)
```

## Tests for Correlation
```
cor(x, y) # conduct tests to if this number is significant

cor.test(x, y) #1. Pearson(default)
cor.test(x, y, method="spearman") #2. Spearman #use rank information
cor.test(x, y, method="kendall") #3. Kendall #use rank information
```
