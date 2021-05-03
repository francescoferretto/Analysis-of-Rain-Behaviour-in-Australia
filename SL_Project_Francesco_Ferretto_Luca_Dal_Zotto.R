##################################################################################
#                       Statistical Learning Project - Mod. B
#                        Francesco Ferretto - Luca Dal Zotto 
#                        "Analysis of the Australian Weather" 
###################################################################################

# Notation for the section, subsection, source: 

#1) ###############################################################################
#                                 Section
###############################################################################

#2) # ------------------- Subsection ----------------------

#3) (source: https://www.google.it/)

###################################################################################
#                             The problem statement
###################################################################################

# This dataset contains over 140,000 daily weather observations from 49 Australian 
# weather stations collected in about 10 years (from 2008 to 2017 with some 
# exception). Just to give a brief introduction, the available variables concern
# the temperature, wind, clouds, humidity, pressure and so on.
# For each observation, the target variable is the amount of rainfall of the 
# following day. This is available both in a continuous quantitative version (for 
# regression tasks) and in a binary categorical version (Did it rain the next day?
# Yes or No) for classification purposes.

# The daily observations are available from http://www.bom.gov.au/climate/data. 
# Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml
# Copyright Commonwealth of Australia 2010, Bureau of Meteorology.

# The dataset used in this project has been downloaded from Kaggle at the link
# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package.

# This dataset is also available in a slightly different version via the R package 
# rattle.data and at https://rattle.togaware.com/weatherAUS.csv.

##################################################################################
#                                    Libraries
##################################################################################

library(VIM) # aggr function -> Na analysis
library(e1071) # to check normality (used once to give the idea)
library(lubridate) # used to extract the month and the quarters from the date
library(pROC) # used for classification metrics
library(ROSE) # used for undersampling
library(naniar) # used for the study of NA distribution 
library(corrplot) # used for the correlation graph
library(dplyr) # DataFrame management  
library(ggplot2) # graphs
library(lmtest) # Durbin-Watson Test for autocorrelation of the residuals 
library(e1071) # kurtosis and skewness 
library(glmnet) # Ridge Regression 
library(leaps) # Best Subset Selection
library(truncreg) # Truncated Normal Model 

###################################################################################
#                       Load the dataset and first observations 
###################################################################################

# ------------------------------ Load the dataset  -------------------------------

rm(list=ls())
# Import the dataset
weather <- read.csv("weatherAUS.csv", header = T)
View(weather)
# Looking at the head of the dataset, in order to have a preview on variables:
head(weather)
dim(weather) # 142193 observations, 24 variables
str(weather) 
# Check for unique values
any(duplicated(weather)) 
# FALSE -> there are no duplicates observations

# -------------- Analysis of the Na distribution in the dataset ------------------ 

# The function complete.cases tells us how many rows don't contain NA
length(which(complete.cases(weather))) 
# less than an half of observations doesn't contain Na values (still a lot: 56420) 

# looking at the distribution of NA values
na.vector <- c()
for (i in 1:length(weather)){
  na.vector[i] <- c( length( which ( is.na(weather[,i]) ) ) )
}
names(na.vector) <- names(weather)
na.vector

# Relative Frequency of Na
rel.freq.na <- 100*na.vector/dim(weather)[1]
rel.freq.na[order(rel.freq.na,decreasing = T)]

plt.rel.freq.na <- barplot(rel.freq.na,
                           ylab="Relative Frequency (%)",
                           main="Relative Frequencies of Na values in the Dataset",
                           col = "orange2",xaxt="n")
text(plt.rel.freq.na, par("usr")[3], labels = names(rel.freq.na), srt = 60, adj = c(1.1,1.1), xpd = TRUE, cex=0.84) 

# Extracting the variables that have more than a certain percentage of NA values
which(rel.freq.na>40) 
which(rel.freq.na>50) # there's nothing that surpass the 50% of NA values 
which(rel.freq.na>30) 
# the 6th, 7th, 18th and 19th variables have more than 30% of Na values
names(weather)[c(6,7,18,19)]
# The corresponding variables are "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"

# We might look at the way in which the fraction of Na behaves across the variables
# that have more than 35% of Na values.
# We can use the function aggr of the package VIM.
# (source: https://rpubs.com/Mentors_Ubiqum/NA_Distribution)
summary(aggr(weather[,which(rel.freq.na>35)])) 

# Comment: lots of observations have all these four variables with Na values. If we 
# give a look at the dataset, we can see that some locations do not have any 
# measurement of these variable. Probably, those weather stations do not have the
# equipment required: measuring Evaporation and the fraction of sky covered with 
# clouds need advanced tools.

# In a first moment, we can discard the observation having at least one Na, since
# the dataset will still be large.

# Alternatively, we may try to impute those missing values using the median (recall
# that in the case of the presence of outliers, it is more convenient to impute the 
# median instead of the mean).

# --- looking at the pattern of the Na values wrt the locations in time ---

# all the variables
vis_miss(weather,warn_large_data = F) 

# only the variables affected by an high fraction of Na values 
vis_miss(weather[,which(rel.freq.na>35)],warn_large_data = F)

# attach the dataframe
attach(weather)
# detach(weather)

# we notice a pattern in the absence of measured value wrt to Locations, evident in the following 
# plot:
gg_miss_fct(x = weather, fct = Location)

# We can see that the presence of variables with high portions of NAs are affected by the presence
# of some cities with no observations at all 

# This leads to another fact, that the absence of NA seems to be (in the majority of the cases)
# related to the absence of the sensor needed to acquire the observation rather than a
# missingness due to the influence of other external factor
# So, we might not have a pattern of missingness related to other variables in the overall
# analysis.

# Our analysis will be split into two ways, on the considerations made about the NAs: 

#     1. Removing the predictors with an high portion of NAs'
#     2. Removing the cities with 100% NAs for at least one variable. 

#  Cities that have at least one variable with 100% portion of NAs are: 
#  Adelaide, Albany, BadgerysCreek, Ballarat, Dartmoor, GoldCoast, Launceston, Newcastle, NorahHead,
#  PearceRAAF, Penrith, Richmond, Walpole, Witchcliffe, Wollongong, Albury, Bandigo, Katherine, MountGinini
#  Nhil, SalmonGuns, Tuggeranong, Uluru 

# Also for WindGustSpeed, WinGustDir, Pressure9am, Pressure3pm there are clusters of 100 % NA 

####################################################################################
#                                      EDA
####################################################################################

# --------------------- Univariate Case ---------------------- 

# ---------------- Rainfall -----------------

# The variable Rainfall measures the amount of rainfall recorded for the day in
# mm, while RainToday is a categorical variable derived from Rainfall in this 
# way: its value is Yes if precipitation (mm) in the 24 hours to 9am exceeds 1mm, 
# otherwise No. Therefore, it is a binary variable.
# In addition, there are the variables RISK_MM and RainTomorrow that simply are the
# two previous variables shifted by one day: indeed, they represent the same
# statistic but relative to the following day. RISK_MM has continuous values, so it
# can be used as response variable for regression purpose, while RainTomorrow is a
# binary categorical variable, so it will be used for binary classification tasks.
# To avoid redundancy, only the distribution of RISK_MM and RainTomorrow are studied.

# Check that the declared relationship between RISK_MM and RainTomorrow is true:
# when RainTomorrow is equal to 'No', RISK_MM <= 1
summary(RISK_MM[RainTomorrow == 'No']) # OK
# When RainTomorrow is equal to 'Yes', RISK_MM > 1
summary(RISK_MM[RainTomorrow == 'Yes']) # OK

# Check that the declared relationship between Rainfall and RainToday is true:
# when RainToday is equal to 'No', Rainfall <= 1
summary(Rainfall[RainToday == 'No']) # OK
# When RainToday is equal to 'Yes', Rainfall > 1
summary(Rainfall[RainToday == 'Yes']) # OK


# OBS: Rainfall has 1406 Na values, while RISK_MM has not
summary(Rainfall)
summary(RISK_MM)

weather[which(is.na(Rainfall)),c("RISK_MM","Rainfall","Location","Date")] 
# We see that these values are distributed 
summary(RISK_MM[which(is.na(Rainfall))-1])
unique(RISK_MM[which(is.na(Rainfall))-1])

# There is nothing strange about these values, so they are lost information that
# we can recover, paying attention to the first observation of each station.
# To make this clearer:

# Indices:   1   2   3   4   ...  n-1  n  
# RISK_MM   0.1 0.3 0.4 0.7      0.5   ~
# Rainfall   ~  0.1 0.3 0.4      0.6  0.5

# ~ are the only values that are not in common. One vector is the other "shifted": 
# for the other observations Rainfall[i] = RISK_MM[i-1], within each location. 

# First, check if some of these NA occur during the first recording day of a 
# station: in this case, we cannot use the previous observation to restore Rainfall

which(Location[which(is.na(Rainfall))] != Location[which(is.na(Rainfall))-1])
# It returns 546 and 919. Check if the station actually changed 
na_rainfall = which(is.na(Rainfall))
Location[na_rainfall[545]] # Williamtown
Location[na_rainfall[546]] # Wollongong
Location[na_rainfall[918]] # Watsonia
Location[na_rainfall[919]] # Dartmoor
# Yes, so let's simply drop these two rows

detach(weather)
weather = weather[-c(na_rainfall[546],na_rainfall[919]),]
attach(weather)

which(Location[which(is.na(Rainfall))] != Location[which(is.na(Rainfall))-1])
# Now we can copy the other NA values of Rainfall from RISK_MM of the previous obs.

RainfallNew = Rainfall
RainfallNew[which(is.na(Rainfall))] = RISK_MM[which(is.na(Rainfall))-1]

summary(RainfallNew)
summary(Rainfall)


# Now let's check the distribution of these variables
table(RainTomorrow)
barplot(table(RainTomorrow), col = 'darkturquoise')
# 110316 No and 31877 Yes


par(mfrow=c(2,2))
hist(RISK_MM, breaks = 100, main = "RISK_MM distribution", col="darkturquoise") 
boxplot(RISK_MM, col="darkturquoise", horizontal = TRUE ) 
# RISK_MM is extremely right skewed. This is due to the very high percentage of days
# with no rain. 

LogRISK = log(RISK_MM+1)
LogRainfall = log(RainfallNew +1)
hist(LogRISK, prob = TRUE, main = "LogRISK distribution", col="lightslateblue")
boxplot(LogRISK, col="lightslateblue", horizontal = TRUE )
par(mfrow=c(1,1))


# We can check the distribution of the rainy days.
RainyDay = RISK_MM[RainTomorrow == 'Yes']
par(mfrow=c(2,2))
hist(RainyDay, breaks = 100, main = "RainyDay distribution", col="darkturquoise") 
boxplot(RainyDay, col="darkturquoise", horizontal = TRUE ) 

LogRainyDay = log(RainyDay)
hist(LogRainyDay, prob = TRUE, main = "LogRainyDay distribution", col="lightslateblue")
boxplot(LogRainyDay, col="lightslateblue", horizontal = TRUE )
par(mfrow=c(1,1))

summary(LogRainyDay)

# Define a function that counts the number of outliers and their percentage.  
NumberOfOutliers <- function(variable) {
  first_quartile = quantile(variable, na.rm = TRUE)[2]
  third_quartile = quantile(variable, na.rm = TRUE)[4]
  iqr = IQR(variable, na.rm = TRUE)
  
  lower = first_quartile - 1.5*iqr
  upper = third_quartile + 1.5*iqr
  
  num_outliers = sum(variable < lower | variable > upper, na.rm = TRUE)
  perc_outliers = 100*num_outliers/length(variable)
  
  return(data.frame(num_outliers,perc_outliers))
}

NumberOfOutliers(LogRainyDay)

# Considering only the rainy days and applying a log-transformation, the distribution
# has still a heavy right tail, but at least the percentage of outliers becomes 
# smaller (~ 0.21%). Note that even if the distribution of the log-transformation is 
# approximately bell shaped in the range of observation, it is not symmetric. 

# In what follows, we will examine the other variable in groups, looking at their
# distribution and their correlation between them and with the response variable.


# ---------------- Temperature ----------------
# MinTemp: the minimum temperature in degrees celsius.
# MaxTemp: the maximum temperature in degrees celsius.
# Temp9am: temperature (degrees C) at 9am.
# Temp3pm: temperature (degrees C) at 3pm.

par(mfrow=c(2,2))
hist(MinTemp, main = "MinTemp distribution", col="blue") # MinTemp Histogram
hist(MaxTemp, main = "MaxTemp distribution", col="red" ) # MaxTemp Histogram
boxplot(MinTemp, col="blue", horizontal = TRUE ) # MinTemp boxplot
boxplot(MaxTemp, col="red", horizontal = TRUE)  # MaxTemp boxplot
par(mfrow=c(1,1))
summary(MinTemp)
summary(MaxTemp)

# Both MinTemp and MaxTemp seem to be approx. bell-shaped. 
# MaxTemp is slightly right skewed and presents more outliers. 

NumberOfOutliers(MinTemp)
NumberOfOutliers(MaxTemp)
# The outliers in MinTemp are about 0.05 %, in MaxTemp about 0.4 %

par(mfrow=c(2,2))
hist(Temp9am, main = "Temp9am distribution", col="deepskyblue") 
hist(Temp3pm, main = "Temp3pm distribution", col="sienna1" ) 
boxplot(Temp9am, col="deepskyblue", horizontal = TRUE ) 
boxplot(Temp3pm, col="sienna1", horizontal = TRUE)  
par(mfrow=c(1,1))

summary(Temp9am)
summary(Temp3pm)

NumberOfOutliers(Temp9am)
NumberOfOutliers(Temp3pm)

# Also Temp9am and Temp3pm are approx. bell shaped and the fraction of outliers 
# is small (~0.04% and ~0.3% resp.). 

# From the shapes of the distributions we can see that there seems to be a 
# tendency of the Temp3pm distribution to the shape of the empirical distribution 
# of MaxTemp, suggesting that the maximum temperature during the day is reached 
# in the afternoon.


# --------------- Wind ----------------
# WindGustDir: the direction of the strongest wind gust in the 24 hours to midnight.
# WindGustSpeed: the speed (km/h) of the strongest wind gust in the 24 hours to midnight.
# WindDir9am: direction of the wind at 9am.
# WindDir3pm: direction of the wind at 3pm.
# WindSpeed9am: wind speed (km/h) averaged over 10 minutes prior to 9am.
# WindSpeed3pm: wind speed (km/h) averaged over 10 minutes prior to 3pm.

par(mfrow=c(3,1))
barplot(table(WindGustDir), col = 'lightblue', main = 'WindGustDir')
barplot(table(WindDir9am), col = 'blue',  main = 'WindDir9am')
barplot(table(WindDir3pm), col = 'darkblue',  main = 'WindDir3pm')
par(mfrow=c(1,1))

levels(WindGustDir)
# It is convenient to convert these features into numerical ones.
# One possible choice is to write the compass directions in cartesian coordinates,
# to keep track of the cyclic behavior of the directions.

CardinalToNumbers <- function(variable) {
  l = length(variable)
  NS = rep(0, l)
  WE = rep(0, l)
  
  directions = c("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
  cosine = c(0, 0.5, sqrt(2)/2, sqrt(3)/2, 1, sqrt(3)/2, sqrt(2)/2, 0.5, 
             0, -0.5, -sqrt(2)/2, -sqrt(3)/2, -1, -sqrt(3)/2, -sqrt(2)/2, -0.5)
  sine = c(1, sqrt(3)/2, sqrt(2)/2, 0.5, 0, -0.5, -sqrt(2)/2, -sqrt(3)/2, 
           -1, -sqrt(3)/2, -sqrt(2)/2, -0.5, 0, 0.5, sqrt(2)/2, sqrt(3)/2)
  
  conversor = matrix(c(cosine,sine), nrow = 2, byrow = TRUE)
  colnames(conversor) = directions
  
  for (i in seq(1,l,length.out = l)){
    
    if (is.na(variable[i])){
      NS[i] = NA
      WE[i] = NA
    }
    else {
      NS[i] = conversor[2, variable[i]]
      WE[i] = conversor[1, variable[i]]
    } 
  }
  
  return(data.frame(NS, WE))
  
}

WindGustDirCoordinate = CardinalToNumbers(WindGustDir)
WindDir9amCoordinate = CardinalToNumbers(WindDir9am)
WindDir3pmCoordinate = CardinalToNumbers(WindDir3pm)

# At this point we have a dataframe of two columns for each wind direction.
# Now, we can multiply these dataframes with the wind speed vectors, in order to 
# create the final features related to the wind measurements.
WindGust = WindGustDirCoordinate * WindGustSpeed
Wind9am = WindDir9amCoordinate * WindSpeed9am
Wind3pm = WindDir3pmCoordinate * WindSpeed3pm

WindGustX = as.numeric(unlist(WindGust[,1]))
WindGustY = as.numeric(unlist(WindGust[,2]))

Wind3pmX = as.numeric(unlist(Wind3pm[,1]))
Wind3pmY = as.numeric(unlist(Wind3pm[,2]))

Wind9amX = as.numeric(unlist(Wind9am[,1]))
Wind9amY = as.numeric(unlist(Wind9am[,2]))


# ---------------- Evaporation ----------------
# The so-called Class A pan evaporation (mm) in the 24 hours to 9am.

par(mfrow=c(2,2))
hist(Evaporation, prob = TRUE, breaks = 100, main = "Evaporation distribution", 
     col="lightyellow")
boxplot(Evaporation, col="lightyellow", horizontal = TRUE )
summary(Evaporation)

NumberOfOutliers(Evaporation)
# This variable has a very long right tail, and presents ~1.4 % outliers on the 
# right-hand side. Let's apply a log-transformation

LogEvaporation = log(Evaporation + 1)
hist(LogEvaporation, prob = TRUE, breaks = 25, main = "LogEvaporation distribution", col="lightyellow")
boxplot(LogEvaporation, col="lightyellow", horizontal = TRUE )
par(mfrow=c(1,1))

summary(LogEvaporation)
NumberOfOutliers(LogEvaporation)
# Now the distribution is more symmetric, and the fraction of outliers is 0.33%


# ---------------- Sunshine ----------------
# The number of hours of bright sunshine in the day.

par(mfrow=c(2,1))
hist(Sunshine, main = "Sunshine distribution", col="lightyellow" )     
boxplot(Sunshine, col="lightyellow", horizontal = TRUE )
par(mfrow=c(1,1))

summary(Sunshine)

# We can say that the variable is bimodal, with one pick in 0 and one in 10. 
# Intuitively, the former corresponds to the rainy days, with no hour of sunshine.


# --------------- Cloud ----------------
# Cloud9am: fraction of sky obscured by cloud at 9am. This feature is measured in 
# "oktas", which are a unit of eighths. It is a discrete 
# variable, since only positive integers up to 8 are admitted. However, from 
# the barplot and the summary we can see that some samples have value equal to 9. 
# that is a complete obscuration of the sky.
# Cloud3pm: same but at 3pm.

par(mfrow=c(2,1))
barplot(table(Cloud9am), main = "Cloud9am distribution", col = 'dodgerblue1')
barplot(table(Cloud3pm), main = "Cloud3pm distribution", col = 'dodgerblue3')
par(mfrow=c(1,1))

summary(Cloud9am)
summary(Cloud3pm)

# We can observe that both these variables are bimodal with one pick in 1 one in 7.
# Finally, keep in mind that these features have a large number of NA's. 

# ------------------ Humidity -------------------
# Humidity9am: humidity (percent) at 9am.
# Humidity3pm: humidity (percent) at 3pm.

par(mfrow=c(2,2))
hist(Humidity9am, main = "Humidity9am distribution", col="lightsteelblue2") 
hist(Humidity3pm, main = "Humidity3pm distribution", col="lightsteelblue4" ) 
boxplot(Humidity9am, col="lightsteelblue2", horizontal = TRUE ) 
boxplot(Humidity3pm, col="lightsteelblue4", horizontal = TRUE)  
par(mfrow=c(1,1))

NumberOfOutliers(Humidity9am)
NumberOfOutliers(Humidity3pm)

# The distribution of Humidity3pm is quite symmetric and it has no outliers.
# The distribution of Humidity9am has a pick corresponding to humidity 100%
# and almost 1 % of the samples are outliers.
# Notice that the mean of Humidity9am is larger than the mean of 
# Humidity3pm, as we could have expected (higher humidity in the morning).


#  ---------- Pressure ---------
# Pressure9am: atmospheric pressure (hpa) reduced to mean sea level at 9am.
# Pressure3pm: atmospheric pressure (hpa) reduced to mean sea level at 3pm.

par(mfrow=c(2,2))
hist(Pressure9am, main = "Pressure9am distribution", col="darkseagreen2") 
hist(Pressure3pm, main = "Pressure3pm distribution", col="darkseagreen3" ) 
boxplot(Pressure9am, col="darkseagreen2", horizontal = TRUE ) 
boxplot(Pressure3pm, col="darkseagreen3", horizontal = TRUE)  
par(mfrow=c(1,1))

NumberOfOutliers(Pressure9am)
NumberOfOutliers(Pressure3pm)

summary(Pressure9am)
summary(Pressure3pm)
# Both distributions are approximately bell-shaped
# The number of outliers is between 0.62 and 0.83 %
# In this case it is crucial to keep in mind the range of the values:
# 980.5 <= Pressure9am <= 1041.0  and  977.1 <= Pressure3pm <= 1039.6
# (It is measured in hpa)


# ----------------- Location ----------------
# The common name of the location of the weather station.

table(Location)
levels(Location)
length(levels(Location))
# There are 49 different weather stations. 

barplot(table(Location), col = 'lightsalmon1', main = 'Observations per location')
# Most of them provided about 3000 days of meteorological recordings, 
# but 3 of them provided approx. one half.

table(Location)[table(Location) < 2000]
# They correspond to the stations Katherine, Nhil and Uluru.
Date[Location=='Katherine'][c(1,1559)]
Date[Location=='Nhil'][c(1,1569)]
Date[Location=='Uluru'][c(1,1521)]

table(Location)['Sydney']
Date[Location=='Sydney'][c(1,3337)]

# We can notice that the weather recordings for those stations start from March
# 2013, while the other stations provided data from February 2008.

# Comment: it is difficult to use the variable in the models, since 49 dummy
# variables would be created, so it should be discarded. In this way, the model
# will be independent from the geographical locations of the data, and this could
# be an advantage.
# A possible suggestion for future works should be the one of comparing the 
# accuracies of the models each one fitted only using the data from one location.
# Another possible idea is to differentiate the cities on the coast and the cities in 
# the inner part of the continent.

# Binary Variable Coast (hand-crafted variable)
IsOnTheCoast<-c(1,1,0,0,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,0,1,1,0,1,1,1,1,
                1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0) # 1 is on the coast, 0 isn't 
names(IsOnTheCoast) <- levels(Location)
Coast = as.factor(IsOnTheCoast[Location]) 
# It expresses the fact that a city is near the coast (<50km from the sea ).

boxplot(LogRISK~Coast, col="lightblue", horizontal = TRUE)


# ----------------- Date -----------------
# It is the date of observation (a Date object in the format Year-Month-Day).

levels(Date)
length(levels(Date))
# 3436 different days

# Let's extract some features from this variable
month = month(as.POSIXlt(Date, format="%Y-%m-%d"))
table(month)
barplot(table(month), col = 'tan2',
        main = 'Observations per month')
# We can see that some months have a smaller number of recordings: in some cases
# this is due to the smaller number of days.

quarters = as.factor(quarter(as.POSIXlt(Date, format="%Y-%m-%d")))
table(quarters)
barplot(table(quarters), col = 'tan3',
        main = 'Observations per quarter')


# Analysis of rainfall in different periods
RainPerMonth = rep(0, 12)
for (m in 1:12){
  RainPerMonth[m] = mean(Rainfall[month==m], na.rm = TRUE)
}

RainPerQuarter = rep(0, 4)
for (q in 1:4){
  RainPerQuarter[q] = mean(Rainfall[quarters==q], na.rm = TRUE)
}

par(mfrow=c(2,1))
barplot(RainPerMonth, col = 'deepskyblue', names.arg = c('Jan', 'Feb', 'Mar',
                                                         'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), 
        main = 'Avg Rainfall per month', ylab = 'mm')
barplot(RainPerQuarter, col = 'dodgerblue', names.arg = c('Q1', 'Q2', 'Q3',
                                                          'Q4'), main = 'Avg Rainfall per quarter', ylab = 'mm')
par(mfrow=c(1,1))


# ------- Study the correlation between these variables -----------

weather = weather[,c("Date","Location","MinTemp","MaxTemp","Temp9am","Temp3pm","Evaporation", "Humidity9am",
                     "Humidity3pm","Pressure9am","Pressure3pm", "Sunshine",
                     "Cloud9am","Cloud3pm","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","WindSpeed9am",
                     "WindSpeed3pm","Rainfall","RainToday","RISK_MM","RainTomorrow")]

corrplot(cor(weather[,-c(which((colnames(weather)=="Date")|
                                 (colnames(weather)=="RainToday")|
                                 (colnames(weather)=="Location")|
                                 (colnames(weather)=="RainTomorrow")|
                                 (colnames(weather)=="WindGustDir")|
                                 (colnames(weather)=="WindDir9am")|
                                 (colnames(weather)=="WindDir3pm") ) ) ],use="complete.obs"))

# We can notice multiple correlations, and, in particular, we have clusters of high correlated variables that share the 
# same "nature", for example, the cluster of variables that measure the temperature in different moments of the day and 
# both the Min and Max Temperature. 

# We might encounter the problem of multicollinearity, due to the presence of variables that add redundant information.

# Define the functions "panel.hist" and "panel.cor" provided in the help 
# of the function "pairs"

# panel.hist function: put histograms on the diagonal
panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan",  ...)  
}

# panel.cor function: put (absolute) correlations on the upper panels,
# with font size proportional to the correlations.

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}


#  -------------------- Temperature ------------------------

X =  data.frame(MinTemp[Location == 'Melbourne'], MaxTemp[Location == 'Melbourne'], 
                Temp9am[Location == 'Melbourne'], Temp3pm[Location == 'Melbourne'], 
                LogRISK[Location == 'Melbourne'])
X = na.omit(X)
colnames(X) <- c("MinTemp","MaxTemp","Temp9am","Temp3pm", "LogRISK")

pairs(X, diag.panel=panel.hist, upper.panel=panel.cor, lower.panel=panel.smooth)
# These four variables are highly correlated. We can keep just one of them.
# We can choose MinTemp, since the number of NA is small and the distribution 
# has a 'nicer' shape.


#  ------------- Evaporation ---------------

# Correlation with MinTemp
X =  data.frame(MinTemp[Location == 'Melbourne'], LogEvaporation[Location == 'Melbourne'], 
                LogRISK[Location == 'Melbourne'])
X = na.omit(X)
colnames(X) <- c("MinTemp","LogEvaporation", "LogRISK")

pairs(X, diag.panel=panel.hist, upper.panel=panel.cor, lower.panel=panel.smooth)
# There is a strong correlation between MinTemp and LogEvaporation.
# Since LogEvaporation has lots of NA, it is convenient to keep MinTemp.


# --------------- Cloud ----------------

# Study the correlation between these variables and Sunshine
X =  data.frame(Cloud9am[Location == 'Melbourne'], Cloud3pm[Location == 'Melbourne'], 
                Sunshine[Location == 'Melbourne'], LogRISK[Location == 'Melbourne'])
X = na.omit(X)
colnames(X) <- c("Cloud9am","Cloud3pm", "Sunshine", "LogRISK")

pairs(X, diag.panel=panel.hist, upper.panel=panel.cor, lower.panel=panel.smooth)
# The correlation between Cloud9am and Cloud3pm is 0.41.
# The correlation of these two variables with Sunshine is pretty high (0.61-0.69).
# A possible choice is to keep only Sunshine, since it has a higher correlation 
# with the response variable, and it has continuous values.


#  ----------------- Wind --------------------

# Study the correlation between these variables
X =  data.frame(WindGustX[Location == 'Melbourne'], WindGustY[Location == 'Melbourne'], 
                Wind9amX[Location == 'Melbourne'], Wind9amY[Location == 'Melbourne'], 
                Wind3pmX[Location == 'Melbourne'], Wind3pmY[Location == 'Melbourne'], 
                LogRISK[Location == 'Melbourne'])
X = na.omit(X)
colnames(X) <- c("WindGustX","WindGustY","Wind9amX","Wind9amY", "Wind3pmX", 
                 "Wind3pmY", "LogRISK")

pairs(X, diag.panel=panel.hist, upper.panel=panel.cor, lower.panel=panel.smooth)
# In this case the choice is not easy. We could keep the X and Y component of 
# Wind9am and Wind3pm, since the correlation with WindGust is quite important: 
# around 0.5 for the X axis and 0.7 for the Y axis.


#  ----------------- Humidity -------------------

# Study the correlation between these variables
X =  data.frame(Humidity9am[Location == 'Melbourne'], Humidity3pm[Location == 'Melbourne'], 
                LogRISK[Location == 'Melbourne'])
X = na.omit(X)
colnames(X) <- c("Humidity9am","Humidity3pm", "LogRISK")

pairs(X, diag.panel=panel.hist, upper.panel=panel.cor, lower.panel=panel.smooth)
# The correlation between Humiduty9am and Humiduty3pm is 0.53.
# However, these two variables could be correlated with the response one,
# so it may be convenient to keep both of them, in a first moment. 


# ----------------- Pressure -------------------

# Study the correlation between these variables
X =  data.frame(Pressure9am[Location == 'Melbourne'], Pressure3pm[Location == 'Melbourne'], 
                LogRISK[Location == 'Melbourne'])
X = na.omit(X)
colnames(X) <- c("Pressure9am","Pressure3pm", "LogRISK")

pairs(X, diag.panel=panel.hist, upper.panel=panel.cor, lower.panel=panel.smooth)
# The correlation between Pressure9am and Pressure3pm is very high: 0.96.
# Therefore, we can keep just one of them.

###########################
# Reflections on this part
########################### 

# ---------------- Bimodality -------------------
# We found different kind of Bimodality: 
# Bimodality can be caused by some trends in time that modify the pattern of the 
# data (in a consistent way), in other words, with seasons changing we observe a 
# different behavior.
# In other cases, Bimodality is caused by the nature of the domain of the variable 
# (e.g. Sunshine).
# This could represent a problem since in some cases the majority 
# of the mass of the overall distribution is put on the one of the extremes of the
# domain (Evaporation, RISK_MM, Sunshine, ...)

# ----------------- Outliers -----------------
# Another problem is the fact that some variables seem to be normally distributed, 
# but qqplots suggest significative deviations from the tails in the majority of 
# the cases, this can be understood with different measures, such as kurtosis 
# (index of concentration of values wrt the mean; for a standard normal is 3) and
# skewness (asymmetry index; when is close to zero indicates an high symmetry).
# (source: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)

# Let's consider for example MinTemp
kurtosis(MinTemp,na.rm = T) # -0.4873134 -> light tails
skewness(MinTemp,na.rm = T) # 0.02389931 -> good symmetry in this case
# These tails are somehow explained by the presence of outliers wrt the overall
# distribution of the variables, since, in most of the cases the tails are "too 
# light" respect to a normal distributed variable. 

# A possible hypothesis on the origins of these outliers could be the fact that we 
# consider different cities. The presence of outliers could be due to the different 
# microclimates per each city. 

# ---------------- Filter the datasets -----------------
# One aspect to keep in mind is the dependency between the samples:
# the dataset basically is a (sequence of) time series, so the observations could not be i.i.d.

# To contain the time dependency, we could consider only one observation every 3.
# Let's create a filter that keeps only the days that are multiple of three
day3 = day(as.POSIXlt(Date, format="%Y-%m-%d"))
DayFilter = day3 %% 3 == 0

# Secondly, we remove the locations with at least one variable with all NAs.
LocRemove = c("PearceRAAF","Dartmoor","Richmond", "Ballarat", "Tuggeranong", 
              "Wollongong", "BadgerysCreek", "Penrith", "NorahHead", "GoldCoast","Adelaide",
              "CoffHarbour","Launceston","Newcastle","Walpole","Witchcliffe","Wollongong",
              "Albury","Bandigo","Katherine","MountGinini","Nhil","SalmonsGums","Tuggeranong","Uluru")

LocationFilter = !(Location %in% LocRemove)

# Moreover, some weather stations are very closed geographically: in some cases,
#           there is a station in the city and one in its airport. For this reason, we may also 
# consider a dataset related just to one city.


################################################################################
#                               Regression models
################################################################################

rm(list=ls())
detach()
par(mfrow=c(1,1))

AllAutomaticProcedure <- function(dataset,model){
  print(summary(model)) # Linear Regression Summary
  hist(model$residuals, col="lightblue", main="Model Residuals' Histogram", xlab="Residuals") # Histogram of the residuals
  print(summary(model$residuals)) # Summary of the residuals of the model 
  cat("Skewness of the distribution of the residuals: ",skewness(model$residuals,na.rm = T)) # Skewness
  cat("  Kurtosis of the distribution of the residuals: ", kurtosis(model$residuals,na.rm = T)) # Kurtosis 
  print(acf(model$residuals, main=" Autocorrelation Function",na.action = na.omit)) # ACF function
  print(lmtest::dwtest(model, data=dataset)) # Durbin Watson Test 
  print(car::vif(model)) # VIF Statistic 
  cat("BIC of the model ",BIC(model))
  cat("   AIC of the model ",AIC(model))
  cat("   R^2 of the model ",summary(model)$adj.r.squared ) # BIC, AIC and R^2 of the model  
  plot(model,4) # Potential Leverage Points
  par(mfrow=c(2,2)) 
  plot(model) # Diagnostics of the residuals 
  par(mfrow=c(1,1))
}

#  ------------------------------- Automatic Procedures -------------------------------

# Option 1: all days

# library(glmnet)
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV11 <- read.csv("WeatherV11.csv", header = T)
str(WeatherV11)
WeatherV11$Coast <- as.factor(WeatherV11$Coast)
WeatherV11$quarters <- as.factor(WeatherV11$quarters)
WeatherV11 = WeatherV11[,-c(which((colnames(WeatherV11)=="RainToday")|
                                    (colnames(WeatherV11)=="RainTomorrow")|
                                    (colnames(WeatherV11)=="X")
) ) ]
WeatherV11 = na.omit(WeatherV11)
attach(WeatherV11)

# In our case the following shrinkage methods were inconclusive for our analysis. 

# y = WeatherV11$LogRISK
# X = model.matrix(LogRISK~.,data=WeatherV11)[,-1]
# Ridge 
# m.ridge <- glmnet(X, y, alpha=0)
# plot(m.ridge,xvar='lambda', xlab=expression(log(lambda)))
# set.seed(3029)
# cv.ridge <- cv.glmnet(X, y, alpha=0) #10 fold cross validation 
# plot(cv.ridge, xlab=expression(log(lambda)))
# BestLambda = cv.ridge$lambda.min 
# m.ridge.best <- glmnet(X, y, alpha=0, lambda=BestLambda)
# m.ridge.best
# 
# Lasso 
# m.lasso <- glmnet(X, y, alpha=1)
# plot(m.lasso, xvar='lambda', xlab=expression(log(lambda)))
# cv.lasso <- cv.glmnet(X, y, alpha=1)
# best.lambda.lasso = cv.lasso$lambda.min
# par(mfrow=c(1,2))
# plot(m.lasso, xvar='lambda', xlab=expression(log(lambda)))
# abline(v=log(best.lambda.lasso), lty=2)
# plot(log(m.lasso$lambda), m.lasso$dev.ratio, type='l',xlab=expression(log(lambda)), ylab='Explained Deviance')
# abline(v=log(best.lambda.lasso), lty=2)

# Best Subset Selection
library(leaps)
regfit.full <- regsubsets(LogRISK~., method="exhaustive", data=WeatherV11, nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
fourpanels <- par(mfrow=c(2,2))

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(15,reg.summary$rss[15], col="red",cex=2,pch=20)

plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(15,reg.summary$adjr2[15], col="red",cex=2,pch=20)

plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(15,reg.summary$cp[15], col="red",cex=2,pch=20)

plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(15,reg.summary$bic[15], col="red",cex=2,pch=20)

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp") 
plot(regfit.full,scale="bic")

# library(e1071)
# Reduced Model 
WV11Automatic <- lm(LogRISK~. -MinTemp -Temp9am -quarters -Temp3pm -Wind3pmX 
                    -Wind3pmY -Wind9amX -Wind9amY -WindGustX -Pressure3pm 
                    -WindGustY ,data=WeatherV11, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11Automatic)

# --------------------- Only rainy days ------------------------- 
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV11 <- read.csv("WeatherV11.csv", header = T)
WeatherV11$Coast <- as.factor(WeatherV11$Coast)
WeatherV11$quarters <- as.factor(WeatherV11$quarters)
attach(WeatherV11)
WeatherV11Rainy = WeatherV11[RainTomorrow=="Yes",]
WeatherV11Rainy = WeatherV11Rainy[,-c(which((colnames(WeatherV11Rainy)=="RainToday")|
                                              (colnames(WeatherV11Rainy)=="RainTomorrow")|
                                              (colnames(WeatherV11Rainy)=="X")
) ) ]
detach()
attach(WeatherV11Rainy)

# Best Subset Selection
regfit.full <- regsubsets(LogRISK~., method="exhaustive", data=WeatherV11Rainy, nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
fourpanels <- par(mfrow=c(2,2))

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(15,reg.summary$rss[15], col="red",cex=2,pch=20)

plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(15,reg.summary$adjr2[15], col="red",cex=2,pch=20)

plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(15,reg.summary$cp[15], col="red",cex=2,pch=20)

plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(8,reg.summary$bic[8], col="red",cex=2,pch=20)

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp") 
plot(regfit.full,scale="bic")

# Reduced Model 
WV11AutomaticRainy <- lm(LogRISK~. -MinTemp -Temp9am -Temp3pm -Wind3pmX -Wind3pmY -Wind9amX
                         -Wind9amY -WindGustX -WindGustY -Humidity9am -LogEvaporation
                         -Pressure9am -Cloud3pm -quarters -Coast,
                         data=WeatherV11Rainy, na.action = na.omit)
AllAutomaticProcedure(WeatherV11Rainy,WV11AutomaticRainy)

# --- Dataset with subset of selected variables according to the EDA ---  

# Opt 1: all days

detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV13 <- read.csv("WeatherV13.csv", header = T)
WeatherV13$Coast <- as.factor(WeatherV13$Coast)
WeatherV13$quarters <- as.factor(WeatherV13$quarters)
WeatherV13 = WeatherV13[,-c(which((colnames(WeatherV13)=="RainToday")|
                                    (colnames(WeatherV13)=="RainTomorrow")|
                                    (colnames(WeatherV13)=="X")
) ) ]
str(WeatherV13)
attach(WeatherV13)

# Best Subset Selection 
library(leaps)
regfit.full <- regsubsets(LogRISK~., method="exhaustive", data=WeatherV13[LogRISK>0,], nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
par(mfrow=c(2,2))

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(14,reg.summary$rss[14], col="red",cex=2,pch=20)

plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(14,reg.summary$adjr2[14], col="red",cex=2,pch=20)

plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(13,reg.summary$cp[13], col="red",cex=2,pch=20)

plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(11,reg.summary$bic[11], col="red",cex=2,pch=20)

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp") 
plot(regfit.full,scale="bic")

# library(e1071)
# Reduced Model 
WV13All <- lm(LogRISK~.,data=WeatherV13, na.action = na.omit) 
AllAutomaticProcedure(WeatherV13,WV13All)

# Reduced Model 
WV13ReducedV1 <- lm(LogRISK~. -quarters,data=WeatherV13[LogRISK>0,], na.action = na.omit) 
AllAutomaticProcedure(WeatherV13,WV13ReducedV1)

# Reduced Model
WV13ReducedV2 <- lm(LogRISK~. -quarters -MinTemp,data=WeatherV13, na.action = na.omit)  
AllAutomaticProcedure(WeatherV13,WV13ReducedV2)

# Opt 2: one observation every three 
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV14 <- read.csv("WeatherV14.csv", header = T)
WeatherV14$Coast <- as.factor(WeatherV14$Coast)
WeatherV14$quarters <- as.factor(WeatherV14$quarters)
WeatherV14 = WeatherV14[,-c(which((colnames(WeatherV14)=="RainToday")|
                                    (colnames(WeatherV14)=="RainTomorrow")|
                                    (colnames(WeatherV14)=="X")
) ) ]
str(WeatherV14)
attach(WeatherV14)

# Best Subset Selection 
regfit.full <- regsubsets(LogRISK~., method="exhaustive", data=WeatherV14, nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
fourpanels <- par(mfrow=c(2,2))

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(14,reg.summary$rss[14], col="red",cex=2,pch=20)

plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)

plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(10,reg.summary$cp[10], col="red",cex=2,pch=20)

plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(9,reg.summary$bic[9], col="red",cex=2,pch=20)

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp") 
plot(regfit.full,scale="bic")

# Reduced Model
WV14Automatic <- lm(LogRISK~.,data=WeatherV14, na.action = na.omit)  
AllAutomaticProcedure(WeatherV14,WV14Automatic)

# --- Dataset with subset of selected variables according to the EDA on Rainy Days ---  

#  Opt 1: all days
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV15 <- read.csv("WeatherV15.csv", header = T)
WeatherV15$Coast <- as.factor(WeatherV15$Coast)
WeatherV15$quarters <- as.factor(WeatherV15$quarters)
WeatherV15 = WeatherV15[,-c(which((colnames(WeatherV15)=="RainToday")|
                                    (colnames(WeatherV15)=="RainTomorrow")|
                                    (colnames(WeatherV15)=="X")
) ) ]
str(WeatherV15)
attach(WeatherV15)

# Best Subset Selection 
library(leaps)
regfit.full <- regsubsets(LogRISK~., method="exhaustive", data=WeatherV15, nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
fourpanels <- par(mfrow=c(2,2))

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(14,reg.summary$rss[14], col="red",cex=2,pch=20)

plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(10,reg.summary$adjr2[10], col="red",cex=2,pch=20)

plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(10,reg.summary$cp[10], col="red",cex=2,pch=20)

plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(6,reg.summary$bic[6], col="red",cex=2,pch=20)

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp") 
plot(regfit.full,scale="bic")

# library(e1071)

# Reduced Model 
WV15Automatic <- lm(LogRISK~.,data=WeatherV15, na.action = na.omit) 
AllAutomaticProcedure(WeatherV15,WV15Automatic)

 # Reduced Model 
WV15ReducedV1 <- lm(LogRISK~. -quarters ,data=WeatherV15, na.action = na.omit)
AllAutomaticProcedure(WeatherV15,WV15ReducedV1)

# Opt 2: one observation every three days 
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV16 <- read.csv("WeatherV16.csv", header = T)
WeatherV16$Coast <- as.factor(WeatherV16$Coast)
WeatherV16$quarters <- as.factor(WeatherV16$quarters)
WeatherV16 = WeatherV16[,-c(which((colnames(WeatherV16)=="RainToday")|
                                    (colnames(WeatherV16)=="RainTomorrow")|
                                    (colnames(WeatherV16)=="X")
) ) ]
str(WeatherV16)
attach(WeatherV16)

# Best Subset Selection 
library(leaps)
regfit.full <- regsubsets(LogRISK~., method="exhaustive", data=WeatherV16, nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
fourpanels <- par(mfrow=c(2,2))

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(14,reg.summary$rss[14], col="red",cex=2,pch=20)

plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(9,reg.summary$adjr2[9], col="red",cex=2,pch=20)

plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(9,reg.summary$cp[9], col="red",cex=2,pch=20)

plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(7,reg.summary$bic[7], col="red",cex=2,pch=20)

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp") 
plot(regfit.full,scale="bic")

# library(e1071)

# WV16All <- lm(LogRISK~.,data=WeatherV16, na.action = na.omit) # Reduced Model 
# AllAutomaticProcedure(WeatherV16,WV16All)

# Reduced Model
# WV16ReducedV1 <- lm(LogRISK~. -quarters ,data=WeatherV16, na.action = na.omit)  
# AllAutomaticProcedure(WeatherV16,WV16ReducedV1)

# Reduced Model 
WV16ReducedV2  <- lm(LogRISK~. -quarters -Coast,data=WeatherV16, na.action = na.omit) 
AllAutomaticProcedure(WeatherV16,WV16ReducedV2)

# ------------- Dataset with One City, with LogRisk>0 -------------

detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV17 <- read.csv("WeatherV17.csv", header = T)
WeatherV17$quarters <- as.factor(WeatherV17$quarters)
WeatherV17 = WeatherV17[,-c(which((colnames(WeatherV17)=="RainToday")|
                                    (colnames(WeatherV17)=="RainTomorrow")|
                                    (colnames(WeatherV17)=="X")|
                                    (colnames(WeatherV17)=="Coast")
) ) ]
attach(WeatherV17)
WeatherV17Positive = WeatherV17[LogRISK>0,]
str(WeatherV17Positive)
detach(WeatherV17)
attach(WeatherV17Positive)

library(leaps)
regfit.full <- regsubsets(LogRISK~., method="exhaustive", data=WeatherV17Positive, nvmax = 15)
summary(regfit.full)
reg.summary <- summary(regfit.full)
fourpanels <- par(mfrow=c(2,2))

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(15,reg.summary$rss[15], col="red",cex=2,pch=20)

plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(15,reg.summary$adjr2[15], col="red",cex=2,pch=20)

plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(12,reg.summary$cp[12], col="red",cex=2,pch=20)

plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(9,reg.summary$bic[9], col="red",cex=2,pch=20)

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp") 
plot(regfit.full,scale="bic")

# library(e1071)
# Reduced Model 
WV17All <- lm(LogRISK~.,data=WeatherV17Positive, na.action = na.omit) 
AllAutomaticProcedure(WeatherV17Positive,WV17All)

# Reduced Model 
WV17ReducedV1 <- lm(LogRISK~. -MinTemp - MaxTemp -Temp9am -LogEvaporation 
                    -Cloud3pm -Cloud9am - LogRainfall ,data=WeatherV17Positive,
                    na.action = na.omit) 
AllAutomaticProcedure(WeatherV17Positive,WV17ReducedV1)

# Reduced Model
WV17ReducedV2 <- lm(LogRISK~.  -MinTemp - MaxTemp -Temp9am -LogEvaporation -Cloud3pm 
                    -Cloud9am  -Pressure9am - LogRainfall -Wind3pmX -Wind9amX,
                    data=WeatherV17Positive, na.action = na.omit)  
AllAutomaticProcedure(WeatherV17Positive,WV17ReducedV2)

#  --------------------------------- Handcrafted Analysis ------------------------------------

# --------------------------- Analysis without NAs --------------------------- 

# ------ Dataset with all transformed variables -------

#  We point out that removing the NAs we will remove automatically the cities that have 100% of missing 
#  observations for the variable: Cloud3pm, Cloud9am, Evaporation, Sunshine, WindGustSPeed, WinGustDir, 
#  Pressure9am, Pressure3pm

# Option 1: all days
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV11 <- read.csv("WeatherV11.csv", header = T)
str(WeatherV11)
WeatherV11$Coast <- as.factor(WeatherV11$Coast)
WeatherV11$quarters <- as.factor(WeatherV11$quarters)
WeatherV11 = WeatherV11[,-c(which((colnames(WeatherV11)=="RainToday")|
                                  (colnames(WeatherV11)=="RainTomorrow")|
                                  (colnames(WeatherV11)=="X")))]
attach(WeatherV11)

# Full Model 
WV11All <- lm(LogRISK~.,data=WeatherV11, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11All)
# Systematic and non linear pattern in the residuals' plot, assumption of normality of the residuals 
# not satisfied,Eteroskedasticity of the residuals - high non-linearity in a neighborhood of zero.
#  The outliers and the leverage points do not represent influencial points (wrt to the Cook's Distance).
# The Autocorrelation is very high.

# Reduced Model V1, removing high VIF variable Temp3pm
WV11ReducedV1 <- lm(LogRISK~.- Temp3pm ,data=WeatherV11, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11ReducedV1)
# The residuals do not change, anyway, we've reduced the complexity of the model, preserving the linear fitting.

anova(WV11All, WV11ReducedV1) # Comparison of the models
# The null hypothesis is (strongly) rejected. We may think that the VIF 
# tolerance should be higher (5-10), as treated in Hastie et Al. (2014). 

# But, we see that the change in the fitting is irrelevant: 
summary(WV11All)$adj.r.squared - summary(WV11ReducedV1)$adj.r.squared
# This model is pretty good, but still present nonlinearities in the residuals, and dependencies. 

# Reduced Model V2, removing high VIF variables Temp9am and Pressure9am: 
WV11ReducedV2 <- lm(LogRISK~. -Temp3pm -Temp9am -Pressure9am, data=WeatherV11, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11ReducedV2)

# Comparison of the models
anova(WV11ReducedV1, WV11ReducedV2) # The null hypothesis is (strongly) rejected.
 
summary(WV11ReducedV1)$r.squared - summary(WV11ReducedV2)$r.squared
# This model is pretty good, but still present nonlinearities in the residuals, and dependencies. 

# Reduced Model V3: We still notice some variables with no significance at all, quarters2 and WindGustX: 
WV11ReducedV3 <- lm(LogRISK~. -Temp3pm -Temp9am -Pressure9am -WindGustX,
                    data=WeatherV11, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11ReducedV3)
anova(WV11ReducedV2, WV11ReducedV3) # The null hypothesis is not rejected. 

# Reduced Model V4: Removing quarters: 
WV11ReducedV4 <- lm(LogRISK~. -Temp3pm -Temp9am -Pressure9am -quarters,
                    data=WeatherV11, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11ReducedV4)
anova(WV11ReducedV2, WV11ReducedV4) # The null hypothesis is (strongly) rejected. 

# Reduced Model V5
WV11ReducedV5 <- lm(LogRISK~. -Temp3pm -Temp9am - Pressure9am -quarters -MaxTemp,data=WeatherV11,
                    na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11ReducedV5)
anova(WV11ReducedV4, WV11ReducedV5) # The null hypothesis is (strongly) rejected.

###########################################################################################################################################################################################################################################
# Reduced Model: Model from the BSS procedure, corrected wrt the removal of Pressure3pm: 
WV11Automatic <- lm(LogRISK~. -MinTemp -Temp9am -quarters -Temp3pm -Wind3pmX -Wind3pmY 
                    -Wind9amX -Wind9amY -WindGustX -Pressure3pm -WindGustY,
                    data=WeatherV11, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11,WV11Automatic)
anova(WV11ReducedV5, WV11Automatic) # Comparison of the models
# The null hypothesis is (strongly) rejected. 
###########################################################################################################################################################################################################################################

# Spotting for potential non-linearity in the predictors    
termplot(WV11Automatic, smooth = panel.smooth, span.smth = 1/4, partial.resid = TRUE)

# Implementation of the Cochrane-Orcutt procedure: 
WV11AllOrc = orcutt::cochrane.orcutt(WV11All) 
WV11AutomaticOrc = orcutt::cochrane.orcutt(WV11Automatic)

summary(WV11AutomaticOrc) # Estimates of the coefficients along with the DW statistic

# Representing graphically the behaviour of the residuals in time:

# Unmodeled residuals
plot(WV11All$residuals[1:3000]~seq(1,3000,by = 1),type="l",col="seagreen") 
lines(smooth.spline(y=WV11All$residuals[1:3000],x=seq(1,3000,by = 1)),lwd=3,col="red")

# Modeled residuals wit AR(1):
plot(WV11AllOrc$residuals~seq(1,56466,1),type ="l",col="skyblue") # Full model
lines(loess.smooth(y=WV11AllOrc$residuals, x=seq(1,56466,1)),col="blue",lwd=2)

plot(WV11AutomaticOrc$residuals~seq(1,56466,1),type ="l",col="pink") # Reduced model automatic 
lines(loess.smooth(y=WV11AutomaticOrc$residuals, x=seq(1,56466,1)),col="red",lwd=2)

# ------------ Only on Rainy Days --------------
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV11 <- read.csv("WeatherV11.csv", header = T)
WeatherV11$Coast <- as.factor(WeatherV11$Coast)
WeatherV11$quarters <- as.factor(WeatherV11$quarters)
attach(WeatherV11)
WeatherV11Rainy = WeatherV11[RainTomorrow=="Yes",]
WeatherV11Rainy = WeatherV11Rainy[,-c(which((colnames(WeatherV11Rainy)=="RainToday")|
                                              (colnames(WeatherV11Rainy)=="RainTomorrow")|
                                              (colnames(WeatherV11Rainy)=="X")
) ) ]
detach()
attach(WeatherV11Rainy)

# Full Model
WV11AllRainyV1 <- lm(LogRISK~. ,data=WeatherV11Rainy, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11Rainy,WV11AllRainyV1)
 
###################################################################################################################################################################################################
# Reduced Model 
WV11AutomaticRainy <- lm(LogRISK~. -MinTemp -Temp9am -Temp3pm -Wind3pmX -Wind3pmY 
                         -Wind9amX -Wind9amY -WindGustX -WindGustY -Humidity9am 
                         -LogEvaporation -Pressure9am -Cloud3pm -quarters -Coast,
                         data=WeatherV11Rainy, na.action = na.omit) 
AllAutomaticProcedure(WeatherV11Rainy,WV11AutomaticRainy)
###################################################################################################################################################################################################

# Spotting for potential non-linearity in the predictors    
termplot(WV11AutomaticRainy, smooth = panel.smooth, span.smth = 1/4, partial.resid = TRUE) 

# Implementation of the Cochrane-Orcutt procedure: 
WV11AutomaticRainyOrc = orcutt::cochrane.orcutt(WV11AutomaticRainy)
summary(WV11AutomaticRainyOrc) # AR(1) model of the errors

# Representing graphically the behaviour of the residuals in time:
plot(WV11AutomaticRainyOrc$residuals~seq(1,12441,1),type ="l",col="skyblue")
lines(loess.smooth(y=WV11AutomaticRainyOrc$residuals, x=seq(1,12441,1)),col="blue",lwd=2)

# Less stringent threshold, only days with LogRISK>0: 
WV11AutomaticRainyNew <- lm(formula = LogRISK ~ . -MinTemp -Temp9am -quarters -Temp3pm 
                            -Wind3pmX -Wind3pmY -Wind9amX -Wind9amY -WindGustX -Pressure3pm 
                            -WindGustY, data = WeatherV11[LogRISK > 0,], na.action = na.omit)
AllAutomaticProcedure(WeatherV11Rainy,WV11AutomaticRainyNew)

# Implementation of the Cochrane-Orcutt procedure: 
WV11AutomaticRainyOrcNew = orcutt::cochrane.orcutt(WV11AutomaticRainyNew)
summary(WV11AutomaticRainyNew) # AR(1) model of the errors

# Option 2: Select one observation every 3 days
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV12 <- read.csv("WeatherV12.csv", header = T)
WeatherV12$Coast <- as.factor(WeatherV12$Coast)
WeatherV12$quarters <- as.factor(WeatherV12$quarters)
WeatherV12 = WeatherV12[,-c(which((colnames(WeatherV12)=="RainToday")|
                                    (colnames(WeatherV12)=="RainTomorrow")|
                                    (colnames(WeatherV12)=="X")
) ) ]
attach(WeatherV12)

WV12All <- lm(LogRISK~.,data=WeatherV12, na.action = na.omit) # Full Model 
AllAutomaticProcedure(WeatherV12,WV12All)
# The dependency is the only difference wrt the previous full model, but still, present.

# --- Dataset just with rainy days ([RainTomorrow == 'Yes'] i.e. more than 1 mm) ---

#  Option 1: all days 
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV15 <- read.csv("WeatherV15.csv", header = T)
WeatherV15$Coast <- as.factor(WeatherV15$Coast)
WeatherV15$quarters <- as.factor(WeatherV15$quarters)
WeatherV15 = WeatherV15[,-c(which((colnames(WeatherV15)=="RainToday")|
                                    (colnames(WeatherV15)=="RainTomorrow")|
                                    (colnames(WeatherV15)=="X")
) ) ]
str(WeatherV15)
attach(WeatherV15)

# Full Model 
WV15All <- lm(LogRISK~. , data=WeatherV15, na.action = na.omit)
AllAutomaticProcedure(WeatherV15,WV15All)
# The model has a better structure than the previous ones, a linear trend in the variance 
# (not so prominent), still the DWtest results are bad, meaning that there's an 
# autocorrelation of the residuals, i.e. a specific trend in the residuals. 
# Another fact is that here we do not have a proper time series, so we should 
# consider a city, or make a multivariate multiple regression model, considering
# the response to be a vector.

# Reduced Model V1: Anyway, we still notice small significance for some variables, and we want to see the effect of the quarters on the 
# residuals' dependency:
WV15ReducedV1 <- lm(LogRISK~. -quarters, data=WeatherV15, na.action = na.omit)
AllAutomaticProcedure(WeatherV15,WV15ReducedV1)
anova(WV15All, WV15ReducedV1) # we do not reject the null hypothesis of the nullity of the coefficient of quarters

# Reduced Model V2
WV15ReducedV2 <- lm(LogRISK~. -quarters -Wind9amX -Humidity9am ,data=WeatherV15, na.action = na.omit) 
AllAutomaticProcedure(WeatherV15,WV15ReducedV2)

anova(WV15ReducedV1,WV15ReducedV2) # we do not reject the null hypothesis of the nullity 
# of the coefficients of Wind9amX and Humidity9am

#  Option 2: Select one observation every 3 days
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV16 <- read.csv("WeatherV16.csv", header = T)
WeatherV16$Coast <- as.factor(WeatherV16$Coast)
WeatherV16$quarters <- as.factor(WeatherV16$quarters)
WeatherV16 = WeatherV16[,-c(which((colnames(WeatherV16)=="RainToday")|
                                    (colnames(WeatherV16)=="RainTomorrow")|
                                    (colnames(WeatherV16)=="X")
) ) ]
str(WeatherV16)
attach(WeatherV16)

 # Full Model
WV16All <- lm(LogRISK~. , data=WeatherV16, na.action = na.omit) 
AllAutomaticProcedure(WeatherV16,WV16All)

 # Reduced Model V1
WV16ReducedV1 <- lm(LogRISK~. -quarters, data=WeatherV16, na.action = na.omit)
AllAutomaticProcedure(WeatherV16,WV16ReducedV1)
anova(WV16All, WV16ReducedV1) # we do not reject the null hypothesis of the nullity of the coefficient of quarters

# Reduced Model V2
WV16ReducedV2 <- lm(LogRISK~. -quarters -Wind9amX -Coast, data=WeatherV16, na.action = na.omit)
AllAutomaticProcedure(WeatherV16,WV16ReducedV2)
anova(WV16ReducedV1,WV16ReducedV2) # we do not reject the null hypothesis of the nullity of the coefficients of Wind9amX and Coast

# Reduced Model V3
WV16ReducedV3 <- lm(LogRISK~. -quarters - Wind9amX -Coast -Wind3pmX, data=WeatherV16, na.action = na.omit) 
AllAutomaticProcedure(WeatherV16,WV16ReducedV3)
anova(WV16ReducedV2,WV16ReducedV3) #  we do not reject the null hypothesis of the nullity of the coefficients of Wind3pmX 

# ------------------ Dataset with all transformed variables - One city -----------------

detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV17 <- read.csv("WeatherV17.csv", header = T)
WeatherV17$quarters <- as.factor(WeatherV17$quarters)
WeatherV17 = WeatherV17[,-c(which((colnames(WeatherV17)=="RainToday")|
                                   (colnames(WeatherV17)=="RainTomorrow")|
                                   (colnames(WeatherV17)=="X")|
                                   (colnames(WeatherV17)=="Coast")
))]
str(WeatherV17)
attach(WeatherV17)

# Full Model
WV17All <- lm(LogRISK~. , data=WeatherV17, na.action = na.omit) 
AllAutomaticProcedure(WeatherV17,WV17All)

# Only days with LogRisk>0
detach()
rm(list=setdiff(ls(), "AllAutomaticProcedure"))
WeatherV17 <- read.csv("WeatherV17.csv", header = T)
WeatherV17$quarters <- as.factor(WeatherV17$quarters)
WeatherV17 = WeatherV17[,-c(which((colnames(WeatherV17)=="RainToday")|
                                    (colnames(WeatherV17)=="RainTomorrow")|
                                    (colnames(WeatherV17)=="X")|
                                    (colnames(WeatherV17)=="Coast")
) ) ]
attach(WeatherV17)
WeatherV17Positive = WeatherV17[LogRISK>0,]
str(WeatherV17Positive)
detach(WeatherV17)
attach(WeatherV17Positive)

library(e1071)
# Full Model
WV17All <- lm(LogRISK~.,data=WeatherV17Positive, na.action = na.omit) 
AllAutomaticProcedure(WeatherV17,WV17All)

# Reduced Model 
WV17ReducedV2 <- lm(LogRISK~. -MinTemp -MaxTemp -Temp9am -LogEvaporation -Cloud3pm 
                    -Cloud9am -Pressure9am -LogRainfall -Wind3pmX -Wind9amX, 
                    data=WeatherV17Positive, na.action = na.omit) 
AllAutomaticProcedure(WeatherV17Positive,WV17ReducedV2)

WV17ReducedV2Orc <- orcutt::cochrane.orcutt(WV17ReducedV2)
summary(WV17ReducedV2Orc)

################################################################################
#                            Classification models
################################################################################

# ---- Procedure 1: dataset with a subset of variables and 1 observation  every 3 days ----

# clean the global environment
detach()
rm(list=ls())

# Load the dataset
WeatherClass <- read.csv("WeatherV14.csv", header = T)
View(WeatherClass)

# remove response for regression and the indices
WeatherClass = WeatherClass[,-c(which((colnames(WeatherClass)=="LogRISK")|
                                        (colnames(WeatherClass)=="X")
) ) ]
WeatherClass = na.omit(WeatherClass) # remove NA values

attach(WeatherClass)
n = dim(WeatherClass)[1]
round(table(RainTomorrow)/n, 3)
# RainTomorrow
# No   Yes 
# 0.771 0.229 

# The dataset is highly unbalanced. If we consider as baseline the trivial 
# classifier which classifies all points with the label of the largest class, 
# it's overall accuracy would be 0.771%. 
# As a first attempt, we won't balance the dataset, therefore it is convenient to 
# consider other metrics for the evaluation of a classifier, such as Specificity 
# and Sensitivity.

# Split the dataset in training and test set (80% train, 20% test)
train_size = 4*n%/%5
set.seed(1)
train <- sample(1:n, train_size)

# The model considered is Logistic Regression 
# Start considering all 13 predictors

mod.full <- glm(RainTomorrow ~ MinTemp + Sunshine + Wind3pmX + Wind9amX + Wind9amY + 
                  Wind3pmY + Humidity3pm + Humidity9am + Pressure3pm + 
                  as.factor(quarters) + as.factor(Coast) +  
                  LogRainfall + RainToday, family = binomial, subset=train)

summary(mod.full)

mod.full$deviance  # 12051.06

# To have a first indication of the precision of the model, let's compute the 
# training accuracy of the full model
logistic.prob <- predict(mod.full, type="response") 
logistic.pred <- rep("No", train_size)
logistic.pred[logistic.prob>0.5] <- "Yes" 

table(logistic.pred, RainTomorrow[train])  # Confusion Matrix
mean(logistic.pred == RainTomorrow[train]) # accuracy = 0.8427515


# Let's perform a Backward Feature Selection, with the aim of reducing the model 
# complexity and possibly overfitting and noise.

### Remove Coast, the predicting variable that seems to be less correlated with 
# the response variable (p-value = 0.698884)
mod12 <- glm(RainTomorrow ~ MinTemp + Sunshine + Wind3pmX + Wind9amX + Wind9amY + 
               Wind3pmY + Humidity3pm + Humidity9am + Pressure3pm + 
               as.factor(quarters) +
               LogRainfall + RainToday, family = binomial,
             subset=train)

summary(mod12)
mod12$deviance  # 12051.21
# The deviance 12051.21 is almost the same -> probably Coast is not relevant. 
# Let's check it with the deviance difference test, using the anova() function.
anova(mod12, mod.full, test="Chisq")
# H0: Coast is not relevant
# p-value = 0.6987 -> fail to reject H0 -> Coast is not relevant


### Remove MinTemp (p-value = 0.338955)
mod11 <- glm(RainTomorrow ~ Sunshine + Wind3pmX + Wind9amX + Wind9amY + 
               Wind3pmY + Humidity3pm + Humidity9am + Pressure3pm + 
               as.factor(quarters) + LogRainfall + RainToday, family = binomial,
             subset=train)

summary(mod11)
mod11$deviance  # 12052.13
# The deviance 12052.13  is almost the same -> probably MinTemp is not relevant
anova(mod11, mod.full, test="Chisq")
# pvalue = 0.5871 -> fail to reject H0 -> MinTemp is not relevant


### Remove RainToday (that is highly correlated with LogRainfall)
mod10 <- glm(RainTomorrow ~ Sunshine + Wind3pmX + Wind9amX + Wind9amY + 
               Wind3pmY + Humidity3pm + Humidity9am + Pressure3pm + 
               as.factor(quarters) + LogRainfall, family = binomial,
             subset=train)

summary(mod10)
# The deviance 12054 is not much bigger -> probably RainToday is not relevant
anova(mod10, mod.full, test="Chisq")
# pvalue = 0.4361 -> fail to reject H0 -> RainToday is not relevant


### Remove Wind3pmY (p-value = 0.025661)
mod9 <- glm(RainTomorrow ~ Sunshine + Wind3pmX + Wind9amX + Wind9amY + 
              Humidity3pm + Humidity9am + Pressure3pm + 
              as.factor(quarters) + LogRainfall, family = binomial,
            subset=train)

summary(mod9)
# The deviance 12059 is not much bigger -> probably Wind3pmY is not relevant
anova(mod9, mod.full, test="Chisq")
# pvalue = 0.1029 -> fail to reject H0 -> Wind3pmY is likely to be irrelevant


# If we try to remove other variables, the deviance test would provide strong 
# evidence for rejection: this means that it is convenient to keep all the remaining
# variables to maintain good generalization capacity.
# For the sake of completeness, let's check the severity of multicollinearity
car::vif(mod9) 
# All values are quite close to 1 (good). The largest are those of Humidity3pm/9am.

# Check the training accuracy of the model with 10 predictors
logistic.prob <- predict(mod9, type="response")
logistic.pred <- rep("No", length(logistic.prob))
logistic.pred[logistic.prob>0.5] <- "Yes"
table(logistic.pred, RainTomorrow[train])
mean(logistic.pred == RainTomorrow[train])  # 0.84299

# The accuracy is slightly higher than the one obtained with the full model. 
# So, removing some features has been a correct choice, since the predicting 
# capacity has not been penalized, and the complexity is lower. However, as said 
# before, the accuracy is not meaningful since the classes are unbalanced.

# Let's tune the threshold to find the optimal Specificity and Sensitivity
roc.out <- roc(RainTomorrow[train], logistic.prob, levels=c("No", "Yes"))

plot(roc.out,  print.auc=TRUE, legacy.axes=TRUE, xlab="False positive rate", 
     ylab="True positive rate")
# AUC = 0.867

# Find the threshold that maximizes the sum of sensitivity and specificity
coords(roc.out, "best")
#  threshold specificity sensitivity
#  0.2042672   0.7738132         0.8

# Compute the corresponding accuracy on the training set
logistic.prob <- predict(mod9, type="response")
logistic.pred <- rep("No", length(logistic.prob))
logistic.pred[logistic.prob>0.2042672] <- "Yes"

table(logistic.pred, RainTomorrow[train]) # Confusion matrix
mean(logistic.pred == RainTomorrow[train]) # accuracy = 0.7797449

# Performance on the test set
test.prob <- predict(mod9, newdata=WeatherClass[-train, ], type="response")
test.pred <- rep("No", length(test.prob))
test.pred[test.prob>0.2042672] <- "Yes"
table(test.pred, RainTomorrow[-train]) # Confusion Matrix

# Accuracy test set
mean(test.pred==RainTomorrow[-train])  # 0.7727489
# Specificity test set 
2411/(2411+780) # 0.7555625
# Sensitivity test set
833/(833+174) # 0.8272095


# ---- Procedure 2: same dataset but balanced using under-sampling ----

# clean the global environment
detach()
rm(list=ls())

# Load the dataset
WeatherClass2 <- read.csv("WeatherV12.csv", header = T)
# remove response for regression, RainToday and the indeces
WeatherClass2 = WeatherClass2[,-c(which((colnames(WeatherClass2)=="LogRISK")|
                                          (colnames(WeatherClass2)=="RainToday")|
                                          (colnames(WeatherClass2)=="X")
) ) ]

WeatherClass2 = na.omit(WeatherClass2) # remove NA values

# Create a balanced dataset with under-sampling
WeatherClass2 <- ovun.sample(RainTomorrow~., data=WeatherClass2, 
                             p=0.5, seed=1, 
                             method="under")$data

# Transform Coast and quarters into factors
WeatherClass2$Coast = as.factor(WeatherClass2$Coast)
WeatherClass2$quarters = as.factor(WeatherClass2$quarters)

table(WeatherClass2$RainTomorrow)
# Now the two classes are almost balanced

n = dim(WeatherClass2)[1]

# Split in training and test set
set.seed(1)
train_size = 4*n%/%5
train <- sample(1:n, train_size)

# creation of model matrix to replace the factors into dummy variables
x <- model.matrix(~., data = WeatherClass2[train, -22])
x.test <- model.matrix(~., data = WeatherClass2[-train,-22])  

# In this case, we will try using Principal Components Analysis to reduce the 
# dimensionality of the model and the covariates dependency
pc <- princomp(x[,-1], cor=TRUE) 
# OBS: -1 to remove the "intercept" column generated automatically with model.matrix
# OBS: with cor = TRUE it uses the correlation matrix
plot(pc, main="Screeplot") # defaul 10 components

# Contributions of the variables to the components - not easy to interpret!
barplot(pc$loadings[,1], cex.names=0.7) # temp + logevaporation + sunshine
barplot(pc$loadings[,2], cex.names=0.7) # mix
barplot(pc$loadings[,3], cex.names=0.7) # y component of wind variables
barplot(pc$loadings[,4], cex.names=0.7) # x component of wind variables
barplot(pc$loadings[,5], cex.names=0.7) # quarters + pressure

# Apply the PCA model to the test set
pc.test <- predict(pc, newdata = x.test)

# Consider only 10 components on the training set
x.pc<-pc$scores[,1:10]

# Create the logistic regression model on the PCA transformed training set
pc1 <- glm(WeatherClass2[train, 'RainTomorrow'] ~., family = binomial, 
           data=as.data.frame(x.pc))
summary(pc1)
# Looking for multicollinearity
car::vif(pc1) # All values are very close to 0 thanks to PCA

# Predictions using default threshold = 0.5
logistic.prob<- predict(pc1, type="response") 
logistic.pred <- rep("No", length(logistic.prob))
logistic.pred[logistic.prob>0.5] <- "Yes"
# Confusion matrix
table(logistic.pred, WeatherClass2[train, 'RainTomorrow']) 
# Accuracy training set 
mean(logistic.pred == WeatherClass2[train, 'RainTomorrow'])  # 0.7768234

# Performance on the test set
test.prob <- predict(pc1, newdata=as.data.frame(pc.test), type="response")
test.pred <- rep("No", length(test.prob))
test.pred[test.prob>0.5] <- "Yes"
# Confusion matrix
table(test.pred, WeatherClass2[-train, 'RainTomorrow'])
# Accuracy test set
mean(test.pred == WeatherClass2[-train, 'RainTomorrow'])  # 0.7685353

# Final remarks: in both cases, the training accuracy is very similar to the test
# accuracy. Probably this is due to the fact that we are using a simple model. 
# More sophisticated techniques could achieve better performances.


# ---- Procedure 3: dataset with one city ----

# clean the global environment
detach()
rm(list=ls())

# Let's consider the dataset with the recordings of just one city
WeatherClass3 <- read.csv("WeatherV17.csv", header = T)
# Remove response for regression
WeatherClass3 = WeatherClass3[,-c(which((colnames(WeatherClass3)=="LogRISK")|
                                          (colnames(WeatherClass3)=="Coast")|
                                          (colnames(WeatherClass3)=="X")
) ) ]
WeatherClass3 = na.omit(WeatherClass3) # remove NA values

attach(WeatherClass3)
n = dim(WeatherClass3)[1]

# Split the dataset in training and test set (80% train, 20% test)
train_size = 4*n%/%5
set.seed(1)
train <- sample(1:n, train_size)

# Model creation: for brevity, we report here just the final model
model <- glm(RainTomorrow ~ MinTemp + Sunshine + Wind3pmX + Wind9amX + Wind9amY + 
               Wind3pmY + Humidity3pm + Pressure3pm + 
               as.factor(quarters) + LogRainfall, family = binomial,
             subset=train)

summary(model)

# Check the training accuracy of the model with 10 predictors
logistic.prob <- predict(model, type="response")
logistic.pred <- rep("No", length(logistic.prob))
logistic.pred[logistic.prob>0.5] <- "Yes"

# Let's tune the threshold to find the optimal Specificity and Sensitivity
roc.out <- roc(RainTomorrow[train], logistic.prob, levels=c("No", "Yes"))

plot(roc.out,  print.auc=TRUE, legacy.axes=TRUE, xlab="False positive rate", 
     ylab="True positive rate")
# AUC = 0.939

# Find the threshold that maximizes the sum of sensitivity and specificity
coords(roc.out, "best")
# threshold specificity sensitivity
# 0.2189115   0.8759086   0.8623482

# Compute the corresponding accuracy on the training set
logistic.prob <- predict(model, type="response")
logistic.pred <- rep("No", length(logistic.prob))
logistic.pred[logistic.prob>0.2189115] <- "Yes"

table(logistic.pred, RainTomorrow[train]) # Confusion matrix
mean(logistic.pred == RainTomorrow[train]) # Accuracy training set = 0.8731405

# Performance on the test set
test.prob <- predict(model, newdata=WeatherClass3[-train, ], type="response")
test.pred <- rep("No", length(test.prob))
test.pred[test.prob>0.2189115] <- "Yes"
# Confusion Matrix
table(test.pred, RainTomorrow[-train])

# Accuracy test set
mean(test.pred==RainTomorrow[-train])  # 0.8479339
# Specificity test set 
411/(411+72) # 0.8509317
# Sensitivity test set
102/(102+20) # 0.8360656


###############################################################################
# Bibliography and Sitography
###############################################################################
# The work done in this project is inspired from the following books and websites:

# Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurelien Geron.

# Introduction to Machine Learning with Python by Andreas C Muller and Sarah Guido.

# Udemy course and Feature Engineering for Machine Learning by Soledad Galli.

# https://www.kaggle.com/prashant111/extensive-analysis-eda-fe-modelling/data )

# Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. 2014. 
# "An Introduction to Statistical Learning: with Applications in R" Springer Publishing Company, Incorporated.

