library(tidyverse) #https://www.tidyverse.org
library(corrplot) #https://cran.r-project.org/web/packages/corrplot/index.html
library(DescTools) #https://www.rdocumentation.org/packages/DescTools/versions/0.99.39
library(cluster) #https://cran.r-project.org/web/packages/cluster/index.html
library(factoextra) #https://www.rdocumentation.org/packages/factoextra/versions/1.0.7

rm(list = ls())

# Preprocessing of the data

# Combining the two datasets into one by suing the attribute type to distinguish
# between the red and white wines
df.red <- read.csv("winequality-red.csv", stringsAsFactors = F, sep = ";")
df.white <- read.csv("winequality-white.csv", stringsAsFactors = F, sep = ";")

df.red$type <- "red"
df.white$type <- "white"

df <- rbind(df.red, df.white)
df$type <- as.factor(df$type)

# Plot of two attributes to see if there is a visual difference between red and
# white wines
ggplot(data = df, aes(x = fixed.acidity, y = pH, color = type)) + geom_point()

# Checking for missing values in the data set
which(is.na(df))

# Correlation plot for all attributes
corrplot(cor(df[, 1:length(df) - 1]))

summary(df)

# Principal components analysis
pca <- prcomp(df[, -length(df)])

summary(pca)

screeplot(pca, type = "l", npcs = 10, main = "Screeplot of the first 10 PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)

cumpro <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 3, col="blue", lty=5)
abline(h = cumpro[3], col="blue", lty=5)

cumpro[3]
pca.df <- as.data.frame(pca$x)

x <- df[,1:length(df) - 1]
y <- df[,length(df)]
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# Clustering

df.class <- df$type

df$type <- NULL

# Deciding on the optimal number of clusters
fviz_nbclust(df, pam, method='silhouette')


# Clustering on the original dataset
pam.res <- pam(df, 2)
fviz_cluster(pam.res, ellipse.type = "norm")+
  theme_minimal()

fviz_silhouette(pam.res)

# Clustering on the PCA data
pam.res <- pam(pca.df[, 1:2], 2)
fviz_cluster(pam.res, ellipse.type = "norm")+
  theme_minimal()

fviz_silhouette(pam.res)


# Outlier detection

lof.res <- LOF(pca.df[, 1:2], 50)
outlier.idxs <- which(lof.res > 3)
df[outlier.idxs, ]

pca.df %>% 
  ggplot(aes(x = PC1, y = PC2)) +
  geom_point(alpha = 0.3) +
  geom_point(data = pca.df[outlier.idxs, ],
             aes(PC1, PC2),
             color = "red",
             size = 2)

# Splitting data into train 
# and test data and scaling them
df$class <- df.class

split <- sample.split(df, SplitRatio = 0.7) 
df.train <- subset(df, split == "TRUE") 
df.test <- subset(df, split == "FALSE") 

df.train.scaled <- scale(df.train[1:(length(df) - 1)]) %>%  as.data.frame()
df.test.scaled <- scale(df.test[1:(length(df) - 1)]) %>%  as.data.frame()

df.train.scaled <- cbind(df.train.scaled, df.train["class"])
df.test.scaled <- cbind(df.test.scaled, df.test["class"])

# Classification

library(rpart.plot) #https://www.rdocumentation.org/packages/rpart.plot/versions/3.0.9
library(e1071) #https://www.rdocumentation.org/packages/e1071/versions/1.7-4
library(caTools) #https://www.rdocumentation.org/packages/caTools/versions/1.17.1
library(caret) #https://www.rdocumentation.org/packages/caret/versions/6.0-86

# Decision tree classifier
tree.classifier <- rpart(class~., df.train.scaled, method = "class")
rpart.plot(tree.classifier)

prediction <- predict(tree.classifier, df.test.scaled, type = "class")

table_mat <- table(df.test$class, prediction)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))

# Naive Bayes classifier
bayes.classifier <- naiveBayes(class~., df.train.scaled)
prediction <- predict(bayes.classifier, df.test.scaled, type = "class")

table_mat <- table(df.test$class, prediction)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))

require(ISLR)

# Logistic regression classifier
glm.classifier <- glm(class~., df.train.scaled, family = "binomial")
glm.prob <- predict(glm.classifier, df.test.scaled, type = "response")
glm.pred <- ifelse(glm.prob > 0.5, "White", "Red")

table_mat <- table(df.test$class, glm.pred)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))

# Regression

# Linear regression for all attributes
lin.mod <- lm(quality ~ ., df.train)
summary(lin.mod)

# Linear regression without two insignificant attributes citric.acid 
# and chlorides
lin.mod <- lm(quality ~ .-citric.acid - chlorides, df.train)
summary(lin.mod)

library(ggpubr) #https://www.rdocumentation.org/packages/ggpubr/versions/0.4.0

# QQ plot
ggqqplot(rstandard(lin.mod), shape=1) +
  ggtitle("") + xlab("Teorethical quantile") +
  ylab("Standardized residual")

predict(lin.mod, df.test)

# Residuals plot
res <- resid(lin.mod)
plot(fitted(lin.mod), res, xlab = "prediction", ylab = "residual")
abline(0,0)

# Density plot
plot(density(res))


