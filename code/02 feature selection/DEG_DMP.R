# Purpose: to perform DEG, DMP analysis with gene expression, DNA methylation dataset using Limma


closeAllConnections()
rm(list=ls())

library(openxlsx)
library(data.table)
library(parallel)
library(ggplot2)
library(pracma)
library(dgof) 
library(limma)

doDEG <- function(input_tsv, lfc_threshold, p_value_threshold, k, type, mode){
  ge_df <- read.csv(input_tsv, header=TRUE, sep="\t")
  rownames(ge_df)<-ge_df$SampleID
  
  print(head(ge_df[1:5, 1:5]))
  print(tail(ge_df[, 1:5], 5))
  
  ge_df_2<-ge_df[ , -which(names(ge_df) %in% c("SampleID"))]

  
  ge_df_2 <- as.data.frame(sapply(ge_df_2, as.numeric))
  rownames(ge_df_2)<-ge_df$SampleID
  print(head(ge_df_2[1:5, 1:5]))
  print(tail(ge_df_2[, 1:5], 5))
  
  
  No_label <- ge_df_2["Label_No", ]
  Tu_label <- ge_df_2["Label_Tu", ]
  
  Tu_label_2 <- Tu_label
  Tu_label_2[Tu_label_2 == '1'] <- 2
  
  label_df <- rbind(No_label, Tu_label_2)
  label <- colSums(label_df)
  label_df <- rbind(label_df, label)
  rownames(label_df) <- c("Label_No", "Label_Tu", "Label_NoTu")
  
  label_list <- unname(unlist(label_df["Label_NoTu",]))
  
  ## remove two row "Label_No", "Label_Tu"
  drow <- c("Label_No", "Label_Tu")
  ge_df_2_rem <- ge_df_2[!rownames(ge_df_2) %in% drow, ]
  
  print("before design matrix")
  print(head(ge_df_2_rem[1:5, 1:5]))
  print(tail(ge_df_2_rem[, 1:5], 5))
  
  
  ## Design matrix
  design <- model.matrix(~ 0 + factor(label_list))
  colnames(design)<-c("Normal", "Tu")
  
  fit  <- lmFit(ge_df_2_rem, design)
  contrast.matrix_NoTu <- makeContrasts(Tu-Normal, levels=design)
  
  fit2_CY <- contrasts.fit(fit, contrast.matrix_NoTu)
  fit2_CY <- eBayes(fit2_CY)
  
  ## output everything for contrast
  n <- nrow(ge_df_2)
  
  ## get p-value by BH
  top1 <- topTable(fit2_CY, coef="Tu - Normal", n=n, adjust = "BH")
  print(top1)
  out_tsv <- sprintf('./[%s %s] Tu %s.csv', type, k, mode)
  # write.table(top1, out_tsv, quote = FALSE, col.names = NA, sep="\t")
  write.csv(top1, out_tsv)
  
  res <- decideTests(fit2_CY, p.value=p_value_threshold, lfc=lfc_threshold)
  summary(res)
  out_tsv <- sprintf('./[%s %s] Tu %s pval(%4.2f)_lfc(%3.1f).csv', type, k, mode, p_value_threshold, lfc_threshold)
  #write.table(res, out_tsv, quote = FALSE, col.names = NA, sep="\t")
  write.csv(res, out_tsv)
  
}



################################################################################################
setwd("......")
dir.create(".......\\results\\k_fold_train_test\\DEG", recursive = TRUE)
setwd(".......\\results\\k_fold_train_test\\DEG")

p_value_threshold=...
lfc_threshold=...

input_tsv <- '../XY_gexp_train_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "train", "DEG")
input_tsv <- '../XY_gexp_train_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "train", "DEG")
input_tsv <- '../XY_gexp_train_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "train", "DEG")
input_tsv <- '../XY_gexp_train_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "train", "DEG")
input_tsv <- '../XY_gexp_train_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "train", "DEG")

input_tsv <- '../XY_gexp_test_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "test", "DEG")
input_tsv <- '../XY_gexp_test_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "test", "DEG")
input_tsv <- '../XY_gexp_test_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "test", "DEG")
input_tsv <- '../XY_gexp_test_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "test", "DEG")
input_tsv <- '../XY_gexp_test_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "test", "DEG")



setwd(".......")
dir.create(".........\\results\\k_fold_train_test\\DMP", recursive = TRUE)
setwd("........\\results\\k_fold_train_test\\DMP")



p_value_threshold=...
lfc_threshold=...

input_tsv <- '../XY_meth_train_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "train", "DMP")
input_tsv <- '../XY_meth_train_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "train", "DMP")
input_tsv <- '../XY_meth_train_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "train", "DMP")
input_tsv <- '../XY_meth_train_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "train", "DMP")
input_tsv <- '../XY_meth_train_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "train", "DMP")


input_tsv <- '../XY_meth_test_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "test", "DMP")
input_tsv <- '../XY_meth_test_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "test", "DMP")
input_tsv <- '../XY_meth_test_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "test", "DMP")
input_tsv <- '../XY_meth_test_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "test", "DMP")
input_tsv <- '../XY_meth_test_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "test", "DMP")

