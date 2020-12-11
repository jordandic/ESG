#import data
#fred
library(Quandl)
require(Sleuth3)
require(mosaic)
require(knitr)
library(PGRdup)
library(tidyverse)
library(sqldf)
library(readxl)
library(quantmod)
library(fastDummies)
library(neuralnet)
library(e1071)
library(forecast)
library(randomForest)
library(rpart)
library(rpart.plot)
library(caret)
library(qpcR)
library(cluster)
library(BBmisc)

################################################### RSEI INITIAL

#setwd("~/Documents/Analytics w:R")

#RSEI data
rsei <- data.frame(read.csv('RSEI data.csv'))
head(rsei)

#data cleaning
rsei <- rsei[,-7]
head(rsei)

colnames(rsei) <- c('RSEI_Score', 'TRIFID', 'NAME', 'STREET', 'CITY','STATE_TERRITORY')

#data preparation/processing
#install.packages('PGRdup')

#rsei$RSEI_Score <- DataClean(rsei$RSEI_Score, fix.comma=T)
#head(rsei)

rsei$RSEI_Score <- as.numeric(gsub(",","",rsei$RSEI_Score))
head(rsei)

sapply(rsei, class)

#summary

rsei_sum <- sqldf('SELECT SUM(RSEI_Score), STATE_TERRITORY FROM rsei GROUP BY STATE_TERRITORY')
head(rsei_sum)

pie(rsei_sum$`SUM(RSEI_Score)`, rsei_sum$STATE_TERRITORY)
barplot(rsei_sum$`SUM(RSEI_Score)` ~ rsei_sum$STATE_TERRITORY)


#loading the data into R 
########################################### RSEI INDUSTRY

RSEI_scores <- data.frame(read_csv("FinalData.csv"))
head(RSEI_scores)

#summary statistics of RSEI scores by industry sector
v <- c(2,4,5,6,7,8,9)

  for(i in v){
    print(favstats(as.numeric(RSEI_scores[,i])))
  }

#sum of scores by industry section
rsei_sum <- aggregate(x=RSEI_scores$`RSEI.Score`, by=list(RSEI_scores$`Industry.Sector`), FUN=sum)
head(rsei_sum)
#changing column names
colnames(rsei_sum)[1] <- c("Industry")
colnames(rsei_sum)[2] <- c("Sum")

#dava visualization
pie(rsei_sum$Sum, rsei_sum$Industry) #pie chart
barplot(rsei_sum$Sum~rsei_sum$Industry) #bar graph



############################################## ETF DATA
# SPY - the market

# FSCHX - chemical 
# PICK - metal fabrication 
# XME - mining
# JJMTF - primary metals
# XTN - Transportation Equipment 
# VIS - miscellaneous Manufacturing 
# XLU - utilities
# XOP - petroleum etf
# XLI - machinery
# XLK - computers and electronic equipment
# VPU - electrical equipment
# PBJ - food 
# PBJ - beverage
# CUT - wood products 
# KOL - coal mining
# BTI - tobacco (BTI, or British American Tobacco, is one of the largest tobacco companies in the US, and it has sufficient data)
# IP - printing (Intl. Paper, or IP, is the largest paper company in the US)
# EVX - hazardous waste

# 18 different industries

library(quantmod)

# market return vector
getSymbols(Symbols = 'SPY')
ar <- data.frame(annualReturn(SPY))
vec <- c(ar[['yearly.returns']])
vec <- vec[1:12]
vec

m_vec <- c()
c <- c(1:18); c
for(i in c){
  m_vec <- append(m_vec, vec)
}
m_vec

dim(dat)
length(m_vec)

############################ regression analysis ################################

# import data
dat <- read_csv('FinalData.csv')
head(dat)

# investigate for exact collinearity
d.cor <- dat[,-c(1,3)]
cor(d.cor)

# normalize
dat.norm <- cbind(data.frame(dat[,1:3]), normalize(data.frame(dat[,4]), method = 'scale'),
                  normalize(data.frame(dat[,5]), method = 'scale'), normalize(data.frame(dat[,6]), method = 'scale'),
                  normalize(data.frame(dat[,7]), method = 'scale'), normalize(data.frame(dat[,8]), method = 'scale'), 
                  normalize(data.frame(dat[,9]), method = 'scale'))
head(dat.norm)

d.norm.cor <- dat.norm[,-c(1,3)]
head(d.norm.cor)
cor(d.norm.cor)

# factorized by industry
dat.norm.fact <- dummy_cols(dat.norm, select_columns = 'Industry.Sector', remove_first_dummy = T)
head(dat.norm.fact)

# factorize non-normalized data
dat.fact <- dummy_cols(data.frame(dat), select_columns = 'Industry.Sector', remove_first_dummy = T)
head(dat.fact)

# further clean data
reg.dat <- dat.norm.fact[,-c(1,3)]
head(reg.dat)

dat.fact <- dat.fact[,-c(1,3)]
head(dat.fact)


############################################## supervised

###### fixed effects model regression ######
# fixed effects model using only factorized data
dat.fact.reg <- lm(dat.fact$Annual.Return ~ ., dat.fact)
summary(dat.fact.reg)

# F-tests against fixed effect factors
reg.reduced.dat.fact <- lm(dat.fact$Annual.Return ~ dat.fact$RSEI.Score + dat.fact$RSEI.Score.Cancer +
                             dat.fact$RSEI.Score.Noncancer + dat.fact$RSEI.Modeled.Hazard + dat.fact$RSEI.Modeled.Pounds +
                             dat.fact$TRI.Pounds, data = dat.fact)
summary(reg.reduced.dat.fact)

anova(dat.fact.reg, reg.reduced.dat.fact)

# F-tests against RSEI and TRI factors
reg.red.env <- lm(dat.fact$Annual.Return ~ dat.fact$Industry.Sector_Chemicals + dat.fact$`Industry.Sector_Coal Mining` +
                    dat.fact$`Industry.Sector_Computers and Electronic Products` + dat.fact$`Industry.Sector_Electric Utilities` +
                    dat.fact$`Industry.Sector_Electrical Equipment` + dat.fact$`Industry.Sector_Fabricated Metals` +
                    dat.fact$Industry.Sector_Food +
                    dat.fact$`Industry.Sector_Hazardous Waste` +
                    dat.fact$Industry.Sector_Machinery +
                    dat.fact$`Industry.Sector_Metal Mining` +
                    dat.fact$`Industry.Sector_Miscellaneous Manufacturing` +
                    dat.fact$Industry.Sector_Petroleum + 
                    dat.fact$`Industry.Sector_Primary Metals` +
                    dat.fact$Industry.Sector_Printing +
                    dat.fact$Industry.Sector_Tobacco +
                    dat.fact$`Industry.Sector_Transportation Equipment` +
                    dat.fact$`Industry.Sector_Wood Products`, data = dat.fact)

summary(reg.red.env)

anova(dat.fact.reg, reg.red.env)

################################ neural network and decision tree ###################################
# training data
dim(dat.bool)
set.seed(2)
dat.bool$Beat <- dat.bool$bool==1
dat.bool$Lost <- dat.bool$bool==0
head(dat.bool)
train <- sample(dat.bool, size = .6*216, replace = F)
head(train)
dim(train)
inv <- as.numeric(c(train[,10])); inv
train <- train[,-10]
head(train)
# validation data
valid <- data.frame(dat.bool[-inv,])
head(valid)


##### neural network #####
# neural network data
train.nn <- train[,-c(8,9)]
valid.nn <- valid[,-c(8,9)]
dim(train.nn)
head(train.nn)
dim(valid.nn)
head(valid.nn)
# neural network
nn <- neuralnet(bool ~ RSEI.Score + RSEI.Score.Cancer + RSEI.Score.Noncancer +
                RSEI.Modeled.Hazard + RSEI.Modeled.Pounds + TRI.Pounds, data = train.nn, linear.output = F, hidden = 8)

plot(nn, rep='best')

nn$weights

pred_v <- predict(nn, newdata= valid.nn)
confusionMatrix(as.factor(ifelse(pred_v > 0.5, "1", "0")), as.factor(valid.nn$bool))

#### decision tree ####
dt <- rpart(train$bool ~ RSEI.Score + RSEI.Score.Cancer + RSEI.Score.Noncancer +
              RSEI.Modeled.Hazard + RSEI.Modeled.Pounds + TRI.Pounds, data = train ,method = "class", cp = -1, minsplit = 1)

# plot tree
prp(dt, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = -10)
length(dt$frame$var[dt$frame$var == "<leaf>"])

# accuracy
head(valid)
head(train)
dt.pred <- predict(dt, valid, type='class')
confusionMatrix(dt.pred, as.factor(valid$bool))

# prune tree
pruned.dt <- prune(dt, cp = 0.04)
prp(pruned.dt, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(pruned.dt$frame$var == "<leaf>", 'gray', 'white')) 
length(pruned.dt$frame$var[pruned.dt$frame$var == "<leaf>"])


# accuracy of pruned tree
dt.pred <- predict(pruned.dt, valid, type='class')
confusionMatrix(dt.pred, as.factor(valid$bool))


##################################### unsupervised

##### K-means clustering ########
# heatmap
set.seed(2)
head(dat)
d <- data.frame(dat.norm)
o <- data.frame(dat)
head(d)
colnames(d) <- c('SubmissionYear', 'AnnualReturn', 'IndustrySector',   
                 'RSEIScore', 'RSEIScoreCancer', 'RSEIScoreNoncancer', 
                 'RSEIModeledHazard',  'RSEIModeledPounds', 'TRIPounds')
colnames(o) <- c('SubmissionYear', 'AnnualReturn', 'IndustrySector',   
                 'RSEIScore', 'RSEIScoreCancer', 'RSEIScoreNoncancer', 
                 'RSEIModeledHazard',  'RSEIModeledPounds', 'TRIPounds')
head(d)
clust.dat <- sqldf('select IndustrySector, sum(RSEIScore) as RSEI, sum(RSEIScoreCancer) as Carcinogen, 
sum(RSEIScoreNoncancer) as NonCancerous, 
sum(RSEIModeledHazard) as ModelHazard, sum(RSEIModeledPounds) as ModelPounds, sum(TRIPounds) as TRI from d group by IndustrySector')
head(clust.dat)
row.names(clust.dat) <- clust.dat$IndustrySector
clust.dat <- clust.dat[,-1]
head(clust.dat)

heatmap(as.matrix(clust.dat), Colv = NA, hclustfun = hclust, 
        col=rev(paste("gray",1:99,sep="")))

# distance matrix
d_mat <- dist(clust.dat, method = "euclidean")

# hierarchical cluster
hc <- hclust(d_mat, method = 'single')
plot(hc, hang = -1, ann = FALSE)

memb <- cutree(hc, h = 5)
memb
memb <- cutree(hc, h = 10)
memb

agnes(d_mat, method = "single")$ac
agnes(d_mat, method = "average")$ac

# robustness test of single method
hc_a <- hclust(d_mat, method = 'average')
plot(hc_a, hang = -1, ann = FALSE)

head(dat.norm)
cor.data <- data.frame(dat.norm[, -c(1,3)])
cor(cor.data)

#### sub-regressions ####
# printing, beverages, coal mining, & tobacco, Computers and electronic products, wood products at level 5 in hierarchical cluster
# normalized data regression
dat.pbctcw <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from d 
where IndustrySector = "Printing" OR IndustrySector = "Beverages" OR IndustrySector = "Coal Mining"
OR IndustrySector = "Tobacco" OR IndustrySector = "Computers and Electronic Products" 
                    OR IndustrySector = "Wood Products"')
head(dat.pbctcw)
dim(dat.pbctcw)
pbctcw.reg <- lm(dat.pbctcw$AnnualReturn ~., data = dat.pbctcw)
summary(pbctcw.reg)

# original data regression
pbctcw <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from o 
where IndustrySector = "Printing" OR IndustrySector = "Beverages" OR IndustrySector = "Coal Mining"
OR IndustrySector = "Tobacco" OR IndustrySector = "Computers and Electronic Products" 
                    OR IndustrySector = "Wood Products"')

pbctcw.r <- lm(pbctcw$AnnualReturn ~., data = pbctcw)
summary(pbctcw.r)

# electrical equipment, hazardous waste at level 5 in hierarchical cluster
# normalized data regression
dat.eh <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from d 
where IndustrySector = "Electrical Equipment" OR IndustrySector = "Hazardous Waste"')
head(dat.eh)
dim(dat.eh)
eh.reg <- lm(dat.eh$AnnualReturn ~., data = dat.eh)
summary(eh.reg) 
# TRI score is a significant factor for these two industries
# according to the heatmap. these two industries are grouped together based on their TRI scores

# original data regression
eh <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from o 
where IndustrySector = "Electrical Equipment" OR IndustrySector = "Hazardous Waste"')
head(eh)
eh.r <- lm(eh$AnnualReturn ~., data = eh)
summary(eh.r) 

confint(eh.r)

# Machinery, Miscellaneous Manufacturing at level 5 in hierarchical cluster
dat.mm <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from d 
where IndustrySector = "Machinery" OR IndustrySector = "Miscellaneous Manufacturing"')
head(dat.mm)
dim(dat.mm)
mm.reg <- lm(dat.mm$AnnualReturn ~., data = dat.mm)
summary(mm.reg) 
# significant results. this entails that RSEI Score is a useful predictor for these two industries 
#heatmap shows that these two industries are grouped together based on their similar RSEI scores

# original data regression
mm <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from o 
where IndustrySector = "Machinery" OR IndustrySector = "Miscellaneous Manufacturing"')
head(mm)
dim(mm)
mm.r <- lm(mm$AnnualReturn ~., data = mm)
summary(mm.r) 

confint(mm.r)

# combined all above sub-industries where at level height=10 in hierarchical cluster
dat.pbctcwmm <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from d 
where IndustrySector = "Printing" OR IndustrySector = "Beverages" OR IndustrySector = "Coal Mining"
OR IndustrySector = "Tobacco" OR IndustrySector = "Computers and Electronic Products" 
                    OR IndustrySector = "Wood Products" OR IndustrySector = "Electrical Equipment" OR IndustrySector = "Hazardous Waste"
                    OR IndustrySector = "Machinery" OR IndustrySector = "Miscellaneous Manufacturing"')

dim(dat.pbctcwmm)
pbctcwmm.reg <- lm(dat.pbctcwmm$AnnualReturn ~., data = dat.pbctcwmm)
summary(pbctcwmm.reg)

# original data regression
pbctcwmm <- sqldf('select AnnualReturn, RSEIScore, RSEIScoreCancer, RSEIScoreNoncancer,
RSEIModeledHazard, RSEIModeledPounds, TRIPounds from o 
where IndustrySector = "Printing" OR IndustrySector = "Beverages" OR IndustrySector = "Coal Mining"
OR IndustrySector = "Tobacco" OR IndustrySector = "Computers and Electronic Products" 
                    OR IndustrySector = "Wood Products" OR IndustrySector = "Electrical Equipment" OR IndustrySector = "Hazardous Waste"
                    OR IndustrySector = "Machinery" OR IndustrySector = "Miscellaneous Manufacturing"')

dim(pbctcwmm)
pbctcwmm.r <- lm(pbctcwmm$AnnualReturn ~., data = pbctcwmm)
summary(pbctcwmm.r)


