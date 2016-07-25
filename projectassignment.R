###Training table;
getwd()
pml_training<-read.csv("pml-training.csv",header=TRUE)
pml_training<-pml_training[,-c(1,2)];

###Testing table;
pml_testing<-read.csv("pml-testing.csv",header=TRUE)
pml_testing<-pml_testing[,-c(1,2)]
str(pml_training);str(pml_testing)
col<-colnames(pml_training)

###the cvtd_timestamp doesn't have the same levels in the training and testing data;
###It's more appropriate to exclude it;
f1<-table(pml_training$cvtd_timestamp)
f2<-table(pml_testing$cvtd_timestamp)


library(lattice);library(ggplot2)
library(caret);

#install.packages("e1071")
library(e1071);
library(caret);library(ggplot2);library(rpart)
##split the training data to be training and validation data;
set.seed(12345)
inTrain<-createDataPartition(y=pml_training$classe,p=2/3,list=FALSE)
new_training<-pml_training[inTrain,]
new_testing<-pml_training[-inTrain,]

na<-apply(sapply(new_training,is.na),2,sum)
new_training1<-new_training[,na<(nrow(new_training)*0.6)]

dim(new_training1)
str(new_training1)
dataNZV<-nearZeroVar(new_training1,saveMetrics=TRUE)
notvarNZV<-colnames(new_training1)[!(dataNZV$nzv)]

###EXcludecvtd_timestamp as it has different factors in the testing data from that in the training data;
notvarNZV<-notvarNZV[notvarNZV!="cvtd_timestamp"]
new_training2<-new_training1[,notvarNZV]
new_testing2<-new_testing[,colnames(new_training2)]

dim(new_training2);dim(new_testing2)
##> dim(new_training2);dim(new_testing2)
##[1] 13083    56
##[1] 6539   56

#modFit<-train(classe~.,method="rpart",data=new_training2)
modFit<-rpart(classe~.,data=new_training2,method="class")

#install.packages("rattle");#install.packages("RGtk2")

#library(RGtk2);library(rattle)

#fancyRpartPlot(modelFit)

confusionMatrix(new_testing2$classe,predict(modFit,new_testing2,type="class"))

##> confusionMatrix(new_testing2$classe,predict(modFit,new_testing2,type="class"))
##Confusion Matrix and Statistics

##Reference
##Prediction    A    B    C    D    E
##A 1680   59   48   26   47
##B   79  927  133  113   13
##C   10   50 1040   38    2
##D   20   29  147  817   59
##E   10   24   35  104 1029

##Overall Statistics

##Accuracy : 0.84            
##95% CI : (0.8309, 0.8488)
##No Information Rate : 0.2751          
##P-Value [Acc > NIR] : < 2.2e-16       

##Kappa : 0.7981          
##Mcnemar's Test P-Value : < 2.2e-16       

##Statistics by Class:

##Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9339   0.8512   0.7413   0.7441   0.8948
##Specificity            0.9620   0.9380   0.9805   0.9531   0.9679
##Pos Pred Value         0.9032   0.7328   0.9123   0.7621   0.8561
##Neg Pred Value         0.9746   0.9693   0.9328   0.9486   0.9773
##Prevalence             0.2751   0.1665   0.2146   0.1679   0.1759
##Detection Rate         0.2569   0.1418   0.1590   0.1249   0.1574
##Detection Prevalence   0.2844   0.1935   0.1743   0.1639   0.1838
##Balanced Accuracy      0.9479   0.8946   0.8609   0.8486   0.9313


###Random Forest;
library(ggplot2);library(randomForest);
modFit_rf<-randomForest(classe~.,data=new_training2,method="class")
confusionMatrix(new_testing2$classe,predict(modFit_rf,new_testing2,type="class"))
##> confusionMatrix(new_testing2$classe,predict(modFit_rf,new_testing2,type="class"))
##Confusion Matrix and Statistics

##Reference
##Prediction    A    B    C    D    E
##A 1860    0    0    0    0
##B    2 1263    0    0    0
##C    0    2 1137    1    0
##D    0    0    3 1067    2
##E    0    0    0    0 1202

##Overall Statistics

##Accuracy : 0.9985          
##95% CI : (0.9972, 0.9993)
##No Information Rate : 0.2848          
##P-Value [Acc > NIR] : < 2.2e-16       

##Kappa : 0.9981          
##Mcnemar's Test P-Value : NA              

##Statistics by Class:

##Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9989   0.9984   0.9974   0.9991   0.9983
##Specificity            1.0000   0.9996   0.9994   0.9991   1.0000
##Pos Pred Value         1.0000   0.9984   0.9974   0.9953   1.0000
##Neg Pred Value         0.9996   0.9996   0.9994   0.9998   0.9996
##Prevalence             0.2848   0.1935   0.1743   0.1633   0.1841
##Detection Rate         0.2844   0.1931   0.1739   0.1632   0.1838
##Detection Prevalence   0.2844   0.1935   0.1743   0.1639   0.1838
##Balanced Accuracy      0.9995   0.9990   0.9984   0.9991   0.9992

###Random Forest Tree method is better than the tree method;
###Use the Random Forest Tree method to segment the data;
selvar<-colnames(new_training2)[colnames(new_training2)!="classe"]
set.seed(12345);
testing<-pml_testing[,selvar]
predtesting<-predict(modFit_rf,testing,type="class")
as.character(predtesting[1])
pml_output <- function (x){
    n <- length(x)
    final.cat<-NULL
    for(i in 1:n){
        final.cat<-rbind(final.cat,c(paste("problem_id_",i,sep=""),as.character(x[i])))
      }
    filename = paste0("final.cat",".txt")
    print(final.cat)
    write.table(final.cat,file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
pml_output(predtesting)

##> pml_output(predtesting)
##[,1]            [,2]
##[1,] "problem_id_1"  "B" 
##[2,] "problem_id_2"  "A" 
##[3,] "problem_id_3"  "B" 
##[4,] "problem_id_4"  "A" 
##[5,] "problem_id_5"  "A" 
##[6,] "problem_id_6"  "E" 
##[7,] "problem_id_7"  "D" 
##[8,] "problem_id_8"  "B" 
##[9,] "problem_id_9"  "A" 
##[10,] "problem_id_10" "A" 
##[11,] "problem_id_11" "B" 
##[12,] "problem_id_12" "C" 
##[13,] "problem_id_13" "B" 
##[14,] "problem_id_14" "A" 
##[15,] "problem_id_15" "E" 
##[16,] "problem_id_16" "E" 
##[17,] "problem_id_17" "A" 
##[18,] "problem_id_18" "B" 
##[19,] "problem_id_19" "B" 
##[20,] "problem_id_20" "B"
