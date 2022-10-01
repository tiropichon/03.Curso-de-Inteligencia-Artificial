# Author: Javier Abascal Carrasco
# Copyright: EOI
# Class: Curso académico 2021
# Script: 4. Machine Learning con R (segunda parte)

setwd("C:/Users/Javie/Documents/1. Trabajo_Universidad/EOI/EOI Online ML las palmas 2021-03-23/Contenidos R/R Code")
# Ejercicio 2 - Clasificación
#######################################################################
# Descargamos el MNIST dataset
# https://drive.google.com/uc?id=0B6E7D59TV2zWYlJLZHdGeUYydlk&export=download

## Cargando el dataset de MNIST
download.file("https://drive.google.com/uc?id=0B6E7D59TV2zWYlJLZHdGeUYydlk&export=download", 
              destfile = "train.csv")
mnist_data<- read.csv("train.csv")
mnist_data$label<-as.factor(mnist_data$label)

# Explorando el dataset
dim(mnist_data)
head(mnist_data,1)

# Fijando los valores a Negro o Blanco
black_and_white <- function(x){
  return(ifelse(x>128,1,0))
}
mnist_data[,2:dim(mnist_data)[2]] = apply(mnist_data[,2:dim(mnist_data)[2]],
                                          2, black_and_white)

# Separamos en conjunto de train & test
n <- nrow(mnist_data)
ntrain <- round(n*0.7)
set.seed(333)
tindex <- sample(n, ntrain)
head(tindex)
length(tindex)
train_mnist <- mnist_data[tindex,]
test_mnist <- mnist_data[-tindex,]

####################################
# Apliquemos un modelo de RandomForest
####################################
install.packages("randomForest")
library(randomForest)
rf_model = randomForest::randomForest(label~.,
                                      data = train_mnist,
                                      type="classification",
                                      importance=TRUE,
                                      proximity=FALSE,
                                      ntree=100)
  
plot(rf_model)
print(rf_model)

# Variable Importance
# https://stats.stackexchange.com/questions/197827/how-to-interpret-mean-decrease-in-accuracy-and-mean-decrease-gini-in-random-fore
randomForest::importance(rf_model)
randomForest::varImpPlot(rf_model, 
           sort=T,
           n.var=10,
           main="Top 10 - Variable Importance")

# Visualicemos el árbol
randomForest::getTree(rf_model, 2, labelVar = TRUE)

# Validemos el modelo en el conjunto de TEST
pred_test_rf = predict(rf_model, test_mnist)
cfm = table(test_mnist$label, pred_test_rf)
cfm

# % acierto
sum(diag(cfm))/3000 *100

####################################
# Apliquemos un modelo de ML complejo de clasificación 
# GBM Gradient Boosting Machine
####################################
install.packages("gbm")
library(gbm)
?gbm
gbm_model = gbm(train_mnist$label~.
                ,data=train_mnist
                ,distribution='multinomial'
                ,n.trees=100
                ,shrinkage=0.05
                ,interaction.depth=5
                ,bag.fraction = 0.6
                ,train.fraction = 0.9
                ,n.minobsinnode = 10
                ,cv.folds = 4
                ,keep.data=TRUE
                ,verbose=TRUE,
                n.cores = 6)

# Variable Importance
head(
  gbm::summary.gbm(gbm_model, 
                 cBars = 10,
                 order=TRUE)
,10)
best_iter <- gbm::gbm.perf(gbm_model, method="cv")

# Validemos el modelo en el conjunto de TEST
pred_test_gbm <- predict(gbm_model, test_mnist, best_iter, type="response")

test_forecast = data.frame(prob_0 = pred_test_gbm[1:3000,1,1],
                 prob_1 = pred_test_gbm[1:3000,2,1],
                 prob_2 = pred_test_gbm[1:3000,3,1],
                 prob_3 = pred_test_gbm[1:3000,4,1],
                 prob_4 = pred_test_gbm[1:3000,5,1],
                 prob_5 = pred_test_gbm[1:3000,6,1],
                 prob_6 = pred_test_gbm[1:3000,7,1],
                 prob_7 = pred_test_gbm[1:3000,8,1],
                 prob_8 = pred_test_gbm[1:3000,9,1],
                 prob_9 = pred_test_gbm[1:3000,10,1])

test_forecast$value_predicted = colnames(test_forecast)[max.col(test_forecast,ties.method="first")]
test_forecast$real_value = test_mnist$label

library(sqldf)
test_forecast = sqldf("select
    *,
    case 
      when value_predicted = 'prob_0' and real_value = 0 then 1
      when value_predicted = 'prob_1' and real_value = 1 then 1
      when value_predicted = 'prob_2' and real_value = 2 then 1
      when value_predicted = 'prob_3' and real_value = 3 then 1
      when value_predicted = 'prob_4' and real_value = 4 then 1
      when value_predicted = 'prob_5' and real_value = 5 then 1
      when value_predicted = 'prob_6' and real_value = 6 then 1
      when value_predicted = 'prob_7' and real_value = 7 then 1
      when value_predicted = 'prob_8' and real_value = 8 then 1
      when value_predicted = 'prob_9' and real_value = 9 then 1
      else 0 end as correct_forecast
  from test_forecast")

#Calculamos la "Confusion Matrix"
table(test_forecast$real_value, test_forecast$value_predicted)
# % acierto
sum(test_forecast$correct_forecast)/3000 *100


# Testing nuestro Handwriting --> http://picresize.com/
# O la modificamos con magick
install.packages("magick")
library(magick)
i <- image_read('number_1.jpeg')
i %>% image_convert(type = 'Grayscale')
i = i %>% image_convert(type = 'Bilevel')
i = image_scale(i, "28x28!")
image_write(i, "imageEOI.jpeg")

install.packages("jpeg")  ## if necessary

library(jpeg)
img <- readJPEG("imageEOI.jpeg")


# 1 & 0 Black & white
# Fijando los valores a Negro o Blanco
black_and_white <- function(x){
  return(ifelse(x>0.5,0,1))
}
img = apply(img, c(1,2), black_and_white)
# Obtenemos la imagen como un vector de 784 columnas en la dirección correcta
install.packages("OpenImageR")
library(OpenImageR)
img = t(img)
img = flipImage(img, mode = "horizontal")
img = t(c(img))

colnames(img) <- colnames(test_mnist[,])[2:ncol(test_mnist)]
img = as.data.frame(img)
handwriting_prediction <- predict(gbm_model, img, best_iter, type="response")

handwriting_prediction = data.frame(prob_0 = handwriting_prediction[1,1,1],
                           prob_1 = handwriting_prediction[1,2,1],
                           prob_2 = handwriting_prediction[1,3,1],
                           prob_3 = handwriting_prediction[1,4,1],
                           prob_4 = handwriting_prediction[1,5,1],
                           prob_5 = handwriting_prediction[1,6,1],
                           prob_6 = handwriting_prediction[1,7,1],
                           prob_7 = handwriting_prediction[1,8,1],
                           prob_8 = handwriting_prediction[1,9,1],
                           prob_9 = handwriting_prediction[1,10,1])

# SUCCESS!!!!! :D con GBM
handwriting_prediction$value_predicted = colnames(handwriting_prediction)[max.col(handwriting_prediction,ties.method="first")]
handwriting_prediction
# Con RandomForest
predict(rf_model, img)

#######################################################################
# Clustering y reducción de dimensionalidad
#######################################################################
install.packages("Rtsne")
library(Rtsne)
install.packages("stats4")
library(stats4)

## Cargando el dataset de MNIST
download.file("https://drive.google.com/uc?id=0B6E7D59TV2zWYlJLZHdGeUYydlk&export=download", 
              destfile = "train.csv")
mnist_data<- read.csv("train.csv") 
mnist_data$label<-as.factor(mnist_data$label)

# Ejcutando un KMEANS with k=10
KM = kmeans(mnist_data[,-1], centers = length(unique(mnist_data$label)))

## Ejecutando una reducción de dimensión con TSNE a 2 dimensiones
tsne <- Rtsne(mnist_data[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)

## Plotting
Labels<-mnist_data$label
colors = rainbow(length(unique(mnist_data$label)))
names(colors) = unique(mnist_data$label)

par(mfrow = c(2,1))
# Con los Labels iniciales
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=mnist_data$label, col=colors[mnist_data$label])

# Con los labels de KMeans
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=KM$cluster, col=colors[KM$cluster])

