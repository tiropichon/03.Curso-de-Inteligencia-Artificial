col = c(
  "nivel_satisfaccion",
  "ultima_eval",
  "num_project",
  "horas_medias",
  "tiempo_en_la_empresa",
  "accidentes_trabajo",
  "abandona",
  "promocion_5_años",
  "departamento",
  "salario"
)

df = read.csv2(file = 'HR_comma_sep.csv', header = TRUE, sep = ',')
colnames(df) <- col
df$nivel_satisfaccion <- as.numeric(df$nivel_satisfaccion)
df$ultima_eval <- as.numeric(df$ultima_eval)

modelo_1 = df[ , c('nivel_satisfaccion', 'ultima_eval', 'abandona')]

library(ggplot)
ggplot(modelo_1, aes(x = nivel_satisfaccion, y = ultima_eval, color = abandona)) + geom_point()

modelo_1_logit = glm(abandona ~ nivel_satisfaccion + ultima_eval, data=modelo_1, family = "binomial")
summary(modelo_1_logit)
exp(coefficients(modelo_1_logit))

# 1 punto nivel de satisfaccion = -98% abandonar
# 1 punto nivel de evaluación = +66% abandonar

prediction = predict(modelo_1_logit, data.frame(nivel_satisfaccion=0 , ultima_eval =1))

# probabilidades de abandonar
exp(prediction)/(1+exp(prediction ))



# Modelo con todo
modelo_2 = df
modelo_2$salario = as.factor(modelo_2$salario)
modelo_2$departamento = as.factor(modelo_2$departamento)
modelo_2_logit = glm(abandona ~ nivel_satisfaccion + ultima_eval, data=modelo_1, family = "binomial")



# Calculo de la precision de los modelos

install.packages("ROCR")
library(ROCR)

prediccion_modelo_1 = prediction(predict(modelo_1_logit, modelo_1, type="response"), modelo_1$abandona)
prediccion_modelo_2 = prediction(predict(modelo_2_logit, modelo_2, type="response"), modelo_2$abandona)

auc_1 = performance(prediccion_modelo_1, measure = "auc")
auc_2 = performance(prediccion_modelo_2, measure = "auc")

auc_1@y.values
auc_2@y.values
