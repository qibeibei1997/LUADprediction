
import numpy as np
import os.path
import csv
import pandas as pd
import argparse
import os
import matplotlib as mpl
# mpl.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# integrate multi-omics datasets
def buildIntegratedDataset_DNN(xy_gxpr, xy_meth, mode):
	print("buildIntegratedDataset")
	xy_gxpr_meth = []

	n_row_g, n_col_g = xy_gxpr.shape
	n_row_m, n_col_m = xy_meth.shape

	# build random index pair set
	idxSet_No = set()
	idxSet_Tu = set()

	NoArr = [1., 0.]
	TuArr = [0., 1.]
	NoCnt = 0
	TuCnt = 0

	for idx_g in range(0, n_row_g):
		label_g = xy_gxpr[idx_g][-2:]
		# print(label_g)
		for idx_m in range(0, n_row_m):
			label_m = xy_meth[idx_m][-2:]
			# print(label_m)

			# normal
			if np.array_equal(label_g, NoArr) and np.array_equal(label_m, NoArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_No.add(integ_idx)
				# print("normal: " + integ_idx)
				NoCnt += 1

			# AD
			if np.array_equal(label_g, TuArr) and np.array_equal(label_m, TuArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_Tu.add(integ_idx)
				# print("ad: " + integ_idx)
				TuCnt += 1

	print("NoCnt: " + NoCnt.__str__())
	print("TuCnt: " + TuCnt.__str__())
	print("size of idxSet_No: " + len(idxSet_No).__str__())
	print("size of idxSet_Tu: " + len(idxSet_Tu).__str__())

	balanced_sample_size = 0;
	if(len(idxSet_No) > len(idxSet_Tu)):
		balanced_sample_size = len(idxSet_Tu)

	if (len(idxSet_Tu) > len(idxSet_No)):
		balanced_sample_size = len(idxSet_No)

	if mode == "balanced":
		print("balanced_sample_size: " + balanced_sample_size.__str__())

		# for normal
		cnt = 0
		for idx in range(len(idxSet_No)):
			idx_str = idxSet_No.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if(cnt >= balanced_sample_size):
				break

		# for AD
		cnt = 0
		for idx in range(len(idxSet_Tu)):
			idx_str = idxSet_Tu.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if (cnt >= balanced_sample_size):
				break

	if mode != "balanced":
		# for normal
		for idx in range(len(idxSet_No)):
			idx_str = idxSet_No.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

		# for AD
		for idx in range(len(idxSet_Tu)):
			idx_str = idxSet_Tu.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)


	print("xy_gxpr_meth: " + len(xy_gxpr_meth).__str__())

	"""
	for idx in range(0, 10):
		xy = xy_gxpr_meth[idx]
		geneSet_str = ";".join(str(x) for x in xy)
		print(geneSet_str)
	"""
	xy_me_ge_values = np.array(xy_gxpr_meth)
	print(xy_me_ge_values.shape)

	return xy_me_ge_values


# load DEG by limma
def getDEG_limma(filename, Thres_lfc, Thres_pval):
	geneSet = set()
	f = open(filename, 'r')
	inCSV = csv.reader(f)
	#print(inCSV)
	header = next(inCSV)  # for header

	for row in inCSV:
		gene = row[0]
		#print(gene)
		logFC = float(row[1])
		Pval = float(row[4]) ## adj p-val : row[5]

		if abs(logFC) >= Thres_lfc and Pval < Thres_pval:
			geneSet.add(gene)

	print("[limma - DEG] Number of gene set: " + str(len(geneSet)))

	return geneSet


# do feature selection with DEG
def applyFeatSel_DEG_intersectGene(infilename, geneSet):
	selected_genelist = ['SampleID']

	for gene in geneSet:
		selected_genelist.append(gene)

	# Label_No	Label_Tu
	selected_genelist.append('Label_No')
	selected_genelist.append('Label_Tu')

	xy_all_df = pd.read_csv(infilename, sep='\t', usecols=selected_genelist)
	xy = xy_all_df.values

	xy_values = xy[:, 1:-2]
	xy_labels = xy[:, -2:]

	# Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
	xy_labels_1_column = []

	NoArr = [1, 0]
	# AD array
	TuArr = [0, 1]
	num_rows, num_cols = xy_labels.shape
	for i in range(num_rows):
		if np.array_equal(xy_labels[i], NoArr):
			xy_labels_1_column.append(0)
		if np.array_equal(xy_labels[i], TuArr):
			xy_labels_1_column.append(1)

	X_embedded = xy_values
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)

	#print(XY_embedded)
	print(XY_embedded.shape)

	return XY_embedded


# load DMG by limma
def getDMG_limma(filename, lfc, pval, probeGene_map):
	geneSet = set()
	cpgSet  = set()
	f = open(filename, 'r')
	inCSV = csv.reader(f)
	#print(inCSV)
	header = next(inCSV)  # for header
	for row in inCSV:
		probe = row[0]
		#print(gene)
		logFC = float(row[1])
		Pvalue = float(row[4]) # adj p-val : row[5]

		if abs(logFC) >=lfc and Pvalue < pval:
			if probe in probeGene_map.keys():
				gene = probeGene_map[probe]
				geneSet.add(gene)
				cpgSet.add(probe)

	print("[limma - DMG] Number of gene set: " + str(len(geneSet)))
	print("[limma - cpG] Number of gene set: " + str(len(cpgSet)))
	# print("geneSet: " + str(geneSet))
	return geneSet


# do feature selection with DMP
def applyFeatSel_DMP_intersectGene(infilename, geneSet, geneCpgSet_map):
	selected_cpglist = ['SampleID']

	for gene, cpgSet in geneCpgSet_map.items():
		if gene in geneSet:
			for cpg in cpgSet:
				selected_cpglist.append(cpg)

	selected_cpglist.append('Label_No')
	selected_cpglist.append('Label_Tu')

	xy_all_df = pd.read_csv(infilename, sep='\t', usecols=selected_cpglist)
	xy = xy_all_df.values ## sampleID, expr + label 2 columns

	xy_tp = np.transpose(xy)
	xy_values = xy[:, 1:-2]
	xy_labels = xy[:, -2:]

	# Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
	xy_labels_1_column = []

	NoArr = [1, 0]
	# AD array
	TuArr = [0, 1]
	num_rows, num_cols = xy_labels.shape
	for i in range(num_rows):
		if np.array_equal(xy_labels[i], NoArr):
			xy_labels_1_column.append(0)
		if np.array_equal(xy_labels[i], TuArr):
			xy_labels_1_column.append(1)

	X_embedded = xy_values
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)

	#print(XY_embedded.shape)
	#print(XY_embedded)

	return XY_embedded


# do feature selection with MI
def applyDimReduction_MI(infilename, num_comp, scatterPlot_fn, mode):
	print("applyDimReduction MI")
	xy = np.genfromtxt(infilename, unpack=True, delimiter='\t', dtype=str)
	xy_tp = np.transpose(xy)
	print("xy_tp: " + str(xy_tp.shape))
	# xy_values = xy_tp[1:, 1:-2]
	xy_labels = xy_tp[1:, -2:]
	xy_labels = xy_labels.astype(np.float64)

	df = pd.read_csv(infilename, encoding="gbk", sep='\t', index_col=0)
	# print(df)
	data = df.values
	data = data[:, :-2]

	target = df["Label_No"]  # ???????????????????????????????????????label
	# target = np.array(target)
	target = target.values
	from minepy import MINE

	from numpy import array

	from sklearn.feature_selection import SelectKBest

	def mic(x, y):
		m = MINE()

		m.compute_score(x, y)

		return (m.mic(), 0.5)  # ?????? K ???????????????????????????????????????????????????

	mic_select = SelectKBest(lambda X, y: tuple(map(tuple, array(list(map(lambda x: mic(x, y), X.T))).T)), k=num_comp)

	X_new = mic_select.fit_transform(data, target)  # k??????????????????????????????????????????

	mic_results_indexs = mic_select.get_support(True)  # ??????

	# print(mic_results_indexs)

	# datat = pd.read_csv(infilename, usecols=mic_results_indexs, encoding="gbk",sep='\t')
	# # print(datat)
	xy_values = X_new
	# print(datat)
	print("xy_values: " + str(xy_values.shape))
	print("xy_labels: " + str(xy_labels.shape))
	XY_embedded = np.append(xy_values, xy_labels, axis=1)
	print(XY_embedded.shape)
	return XY_embedded


# do feature selection with t-SNE ??????t-SNE??????????????????
def applyDimReduction_TSNE(infilename, num_comp, scatterPlot_fn, mode):
	print("\napplyDimReduction t-SNE")
	xy = np.genfromtxt(infilename, unpack=True, delimiter='\t', dtype=str)
	# xy =
	xy_tp = np.transpose(xy)
	print("xy_tp: " + str(xy_tp.shape))
	xy_featureList = xy_tp[0, 1:]
	#print(xy_featureList)

	xy_values = xy_tp[1:, 1:-2]
	xy_labels = xy_tp[1:, -2:]

	xy_values = xy_values.astype(np.float32)
	xy_labels = xy_labels.astype(np.float32)

	# Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
	xy_labels_1_column = []

	NoArr = [1, 0]
	# AD array
	TuArr = [0, 1]
	num_rows, num_cols = xy_labels.shape
	for i in range(num_rows):
		if np.array_equal(xy_labels[i], NoArr):
			xy_labels_1_column.append(0)
		if np.array_equal(xy_labels[i], TuArr):
			xy_labels_1_column.append(1)

	#print("xy_tp row: " + len(xy_tp).__str__() + "\t" + " col: " + len(xy_tp[0]).__str__())
	#print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
	#print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

	print("xy_values: " + str(xy_values.shape))
	print("xy_labels: " + str(xy_labels.shape))

	X_embedded = TSNE(n_components=num_comp, method='exact').fit_transform(xy_values)
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)
	print("XY_embedded: " + XY_embedded.shape.__str__())

	return XY_embedded


# split input data into x and y
def partitionTrainTest_ML_for_CV_DNN(xy_me_ge_values):
	np.random.shuffle(xy_me_ge_values)

	x_data_List = []
	y_data_List = []

	colSize = len(xy_me_ge_values[0])

	for i in range(len(xy_me_ge_values)):
		x_tmpRow = xy_me_ge_values[i, 1:colSize - 2]
		y_tmpRow = xy_me_ge_values[i, colSize - 1:colSize]

		x_data_List.append(x_tmpRow)
		y_data_List.append(y_tmpRow)

	return np.array(x_data_List), np.array(y_data_List)


# perform conventional machine learning
output_dir_path = r"......\results\k_fold_train_test_results"
if not os.path.exists(output_dir_path): os.mkdir(output_dir_path)

def doMachineLearning_single_Kfold(xy_train, xy_test, k):
	print("doMachineLearning_single")
	# outfilename = output_dir_path + "/[" + str(k) + "]f1_score.tsv"
	# fout = open(outfilename, 'w')
	# fout.write("f1_score\n")

	x_train, y_train = partitionTrainTest_ML_for_CV_DNN(xy_train)
	x_test, y_test = partitionTrainTest_ML_for_CV_DNN(xy_test)

	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)

	"""
	sc = StandardScaler()
	sc.fit(x_train)
	x_train = sc.transform(x_train)
	sc.fit(x_test)
	x_test = sc.transform(x_test)
	"""
	from sklearn.metrics import f1_score
	print("## Random Forest")
	###############################################################################################
	# define a model #rdf_clf.fit(x_data_std, y_data)
	rdf_clf = RandomForestClassifier(criterion='entropy', oob_score=True, n_estimators=100, n_jobs=-1, random_state=0, max_depth=6)
	rdf_clf.fit(x_train, y_train.ravel())

	predicted = rdf_clf.predict(x_test)
	test_acc = accuracy_score(y_test, predicted)
	f1_scores = f1_score(y_test, predicted, average="binary")
	training_acc = accuracy_score(y_train, rdf_clf.predict(x_train))
	roc_auc = metrics.roc_auc_score(y_test.ravel(), predicted.ravel())

	print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
	print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
	print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))
	print(str(k) + "-fold f1_score:" + "\t" + str(f1_scores))
	# fout.write(str(k)+"-fold training accuracy:"+ "\t" + str(training_acc) + "\n" +
	# 			str(k) + "-fold test accuracy:" + "\t" + str(test_acc) +"\n" +
	#            str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc) + "\n" +
	#            str(k) + "-fold f1_score:" + "\t" + str(f1_scores) + "\n")
	rd_acc = f1_scores
	rd_auc = roc_auc

	# KNN ###############################################################################################
	print("## KNN")
	###############################################################################################
	# define a model #rdf_clf.fit(x_data_std, y_data)
	knn_clf = KNeighborsClassifier(n_neighbors=10)
	knn_clf.fit(x_train, y_train.ravel())

	predicted = knn_clf.predict(x_test)
	test_acc = accuracy_score(y_test, predicted)


	f1_scores = f1_score(y_test, predicted,average = "binary")
	training_acc = accuracy_score(y_train, knn_clf.predict(x_train))
	roc_auc = metrics.roc_auc_score(y_test.ravel(), predicted.ravel())

	print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
	print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
	print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))
	print(str(k) + "-fold f1_score:" + "\t" + str(f1_scores))
	# fout.write(str(k) + "-fold training accuracy:" + "\t" + str(training_acc) + "\n" +
	#            str(k) + "-fold test accuracy:" + "\t" + str(test_acc) +"\n" +
	#            str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc) + "\n" +
	#            str(k) + "-fold f1_score:" + "\t" + str(f1_scores) + "\n")
	knn_acc = f1_scores
	knn_auc = roc_auc
	###############################################################################################


	# naive bayesian classifier
	print("## naive bayesian")
	###############################################################################################
	# define a model #rdf_clf.fit(x_data_std, y_data)
	gnb_clf = GaussianNB()
	gnb_clf.fit(x_train, y_train.ravel())

	predicted = gnb_clf.predict(x_test)
	test_acc = accuracy_score(y_test, predicted)
	f1_scores = f1_score(y_test, predicted, average="binary")
	training_acc = accuracy_score(y_train, gnb_clf.predict(x_train))
	roc_auc = metrics.roc_auc_score(y_test.ravel(), predicted.ravel())

	print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
	print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
	print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))
	print(str(k) + "-fold f1_score:" + "\t" + str(f1_scores))
	# fout.write(str(k) + "-fold training accuracy:" + "\t" + str(training_acc) + "\n" +
	#            str(k) + "-fold test accuracy:" + "\t" + str(test_acc) +"\n" +
	#            str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc) + "\n" +
	#            str(k) + "-fold f1_score:" + "\t" + str(f1_scores) + "\n")
	nb_acc = f1_scores
	nb_auc = roc_auc
	###############################################################################################
	# fout.close()
	return rd_acc, rd_auc, knn_acc, knn_auc, nb_acc, nb_auc


# get table between probe and gene
def getProbeGeneMap(mapTableFile,input_file):
	final_col_list = ['ID', "Symbol", "TSS_Coordinate"]
	TSSlist = ['TSS1500', 'TSS200']

	selected_cols = ["ID", "Symbol", "TSS_Coordinate"]
	annot_df = pd.read_csv(mapTableFile, sep='\t', usecols=selected_cols)

	f = open(input_file)
	dmp_df = pd.read_csv(f)
	#print(dmp_df)

	dmp_annot_df = pd.merge(dmp_df, annot_df, on='ID')
	#print(dmp_annot_df)

	dmp_annot_df = dmp_annot_df[final_col_list]
	#print(dmp_annot_df)
	dmp_annot_df['Symbol'] = dmp_annot_df['Symbol'].str.split(';').str[0]
	# dmp_annot_df = dmp_annot_df[dmp_annot_df["UCSC_RefGene_Group"].str.contains('|'.join(TSSlist), na=False)]

	dmp_annot_df.columns = ['ID', 'gene_symbol', 'TSS']

	cpg_geneSymbol_dict = dict(zip(dmp_annot_df['ID'], dmp_annot_df['gene_symbol']))
	print("??????????????????????????????????????? cpg_geneSymbol_dict: " + str(len(cpg_geneSymbol_dict.keys())))
	# print(cpg_geneSymbol_dict)
	return cpg_geneSymbol_dict


# load DEG, DMG by limma
def load_DEG_DMG(filePath, lfc, pval, mode, mapTableFile):
	geneSet = set()
	geneCpgSet_map = {}

	if mode == "DEG":
		degSet = getDEG_limma(filePath, lfc, pval)
		geneSet = degSet

	if mode == "DMP":
		probeGene_map = getProbeGeneMap(mapTableFile, filePath)
		dmgSet = getDMG_limma(filePath, lfc, pval, probeGene_map)  ## should be changed
		geneSet = dmgSet

		for p, g in probeGene_map.items():
			if g in geneCpgSet_map.keys():
				pset = geneCpgSet_map[g]
				pset.add(p)
				geneCpgSet_map[g] = pset
			else:
				pset = set()
				pset.add(p)
				geneCpgSet_map[g] = pset

	return geneSet, geneCpgSet_map


# main
def main(args):
	print("Machine Learning approach")
	input_dir = args.input  # ./results/k_fold_train_test
	output_dir = args.output  # ./results/k_fold_train_test_results

	for j in range(0, 2):
		if j == 0:
			mode = "balanced"  # "unbalanced"
		else:
			mode = "unbalanced"  # "unbalanced"

		print("mode: " + mode)

		## number of features
		num_deg_ge_List = []
		num_dmg_me_List = []

		num_tsne_ge_List = []
		num_tsne_me_List = []

		num_mi_ge_List = []
		num_mi_me_List = []

		## results
		deg_dmg_ge_me_DNN_acc_List = []

		deg_ge_RF_acc_List = []
		deg_ge_RF_auc_List = []
		dmg_me_RF_acc_List = []
		dmg_me_RF_auc_List = []
		deg_dmg_ge_me_RF_acc_List = []
		deg_dmg_ge_me_RF_auc_List = []

		deg_ge_KNN_acc_List = []
		deg_ge_KNN_auc_List = []
		dmg_me_KNN_acc_List = []
		dmg_me_KNN_auc_List = []
		deg_dmg_ge_me_KNN_acc_List = []
		deg_dmg_ge_me_KNN_auc_List = []

		deg_ge_NB_acc_List = []
		deg_ge_NB_auc_List = []
		dmg_me_NB_acc_List = []
		dmg_me_NB_auc_List = []
		deg_dmg_ge_me_NB_acc_List = []
		deg_dmg_ge_me_NB_auc_List = []

		tsne_ge_RF_acc_List = []
		tsne_ge_RF_auc_List = []
		tsne_me_RF_acc_List = []
		tsne_me_RF_auc_List = []
		tsne_ge_me_RF_acc_List = []
		tsne_ge_me_RF_auc_List = []

		tsne_ge_KNN_acc_List = []
		tsne_ge_KNN_auc_List = []
		tsne_me_KNN_acc_List = []
		tsne_me_KNN_auc_List = []
		tsne_ge_me_KNN_acc_List = []
		tsne_ge_me_KNN_auc_List = []

		tsne_ge_NB_acc_List = []
		tsne_ge_NB_auc_List = []
		tsne_me_NB_acc_List = []
		tsne_me_NB_auc_List = []
		tsne_ge_me_NB_acc_List = []
		tsne_ge_me_NB_auc_List = []

		mi_ge_RF_acc_List = []
		mi_ge_RF_auc_List = []
		mi_me_RF_acc_List = []
		mi_me_RF_auc_List = []
		mi_ge_me_RF_acc_List = []
		mi_ge_me_RF_auc_List = []

		mi_ge_KNN_acc_List = []
		mi_ge_KNN_auc_List = []
		mi_me_KNN_acc_List = []
		mi_me_KNN_auc_List = []
		mi_ge_me_KNN_acc_List = []
		mi_ge_me_KNN_auc_List = []

		mi_ge_NB_acc_List = []
		mi_ge_NB_auc_List = []
		mi_me_NB_acc_List = []
		mi_me_NB_auc_List = []
		mi_ge_me_NB_acc_List = []
		mi_ge_me_NB_auc_List = []

		## for each k
		for k in range(1, 6):
			print("\n\nK: " + str(k))

			# make directories
			# table 1
			dirPath_table1_ge = output_dir + "/k_" + str(k) + "/table_1/genExpr"
			if not os.path.exists(dirPath_table1_ge): os.mkdir(dirPath_table1_ge)
			dirPath_table1_me = output_dir + "/k_" + str(k) + "/table_1/meth"
			if not os.path.exists(dirPath_table1_me): os.mkdir(dirPath_table1_me)

			# table 2
			dirPath_table2_geme = output_dir + "/k_" + str(k) + "/table_2/genExpr_meth"
			if not os.path.exists(dirPath_table2_geme): os.mkdir(dirPath_table2_geme)

			# table 3
			dirPath_table3_deg = output_dir + "/k_" + str(k) + "/table_3/DEG"
			if not os.path.exists(dirPath_table3_deg): os.mkdir(dirPath_table3_deg)
			dirPath_table3_dmg = output_dir + "/k_" + str(k) + "/table_3/DMG"
			if not os.path.exists(dirPath_table3_dmg): os.mkdir(dirPath_table3_dmg)
			dirPath_table3_deg_dmg = output_dir + "/k_" + str(k) + "/table_3/DEG_DMG"
			if not os.path.exists(dirPath_table3_deg_dmg): os.mkdir(dirPath_table3_deg_dmg)

			#  table 4
			dirPath_table4_deg_dmg = output_dir + "/k_" + str(k) + "/table_4/DEG_DMG"
			if not os.path.exists(dirPath_table4_deg_dmg): os.mkdir(dirPath_table4_deg_dmg)


			# ExpResult 1. mi, tSNE + ML algorithms with same # of reduced features for each GE, Meth
			# ExpResult 2. mi, tSNE + ML algorithms with same # of reduced features by integrating GE, Meth
			# ExpResult 3. DEG, DMG, DEG + DMG + ML algorithms
			# ExpResult 4. DE G + DMG + DNN

			print("\n\nExperiment 3. DEG, DMP + ML")
			# ExpResult 3. DEG, DMG, DEG + DMG + ML algorithms
			################################################################################################################
			# make training, test dataset by our feature selection approach
			thresh_pval_me = ...
			thresh_pval_ge = ...
			thresh_lfc_me = ...
			thresh_lfc_ge = ...

			mapTableFile = ".....\GPL8490-65.txt"

			# training
			# load DEG, DMG for
			degSet, _ = load_DEG_DMG(input_dir + "/DEG/[train " + str(k) + "] Tu DEG.csv", thresh_lfc_ge, thresh_pval_ge, "DEG", mapTableFile)
			dmgSet, geneCpgSet_map = load_DEG_DMG(input_dir + "/DMP/[train " + str(k) + "] Tu DMP.csv", thresh_lfc_me, thresh_pval_me, "DMP", mapTableFile)

			its_geneSet = degSet & dmgSet

			train_xy_gxpr = applyFeatSel_DEG_intersectGene(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", its_geneSet)
			train_xy_meth = applyFeatSel_DMP_intersectGene(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", its_geneSet, geneCpgSet_map)

			print("train_xy_gxpr: " + str(train_xy_gxpr.shape))
			print("train_xy_meth: " + str(train_xy_meth.shape))

			print("\n")
			print("# feature (gene expression):" + str(train_xy_gxpr.shape[1]))
			print("# feature (DNA methylation):" + str(train_xy_meth.shape[1]))
			print("\n")

			num_deg_ge = train_xy_gxpr.shape[1] - 2
			num_dmg_me = train_xy_meth.shape[1] - 2

			test_xy_gxpr = applyFeatSel_DEG_intersectGene(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", its_geneSet)
			test_xy_meth = applyFeatSel_DMP_intersectGene(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", its_geneSet, geneCpgSet_map)

			print("test_xy_gxpr: " + str(test_xy_gxpr.shape))
			print("test_xy_meth: " + str(test_xy_meth.shape))

			train_xy_gxpr_meth = buildIntegratedDataset_DNN(train_xy_gxpr, train_xy_meth, mode) # "unbalanced"
			test_xy_gxpr_meth = buildIntegratedDataset_DNN(test_xy_gxpr, test_xy_meth, mode)

			print("train_xy_gxpr_meth: " + str(train_xy_gxpr_meth.shape))
			print("test_xy_gxpr_meth: " + str(test_xy_gxpr_meth.shape))

			deg_ge_RF_acc, deg_ge_RF_auc, deg_ge_KNN_acc, deg_ge_KNN_auc, deg_ge_NB_acc, deg_ge_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr, test_xy_gxpr, k)
			dmg_me_RF_acc, dmg_me_RF_auc, dmg_me_KNN_acc, dmg_me_KNN_auc, dmg_me_NB_acc, dmg_me_NB_auc = doMachineLearning_single_Kfold(train_xy_meth, test_xy_meth, k)
			deg_dmg_ge_me_RF_acc, deg_dmg_ge_me_RF_auc, deg_dmg_ge_me_KNN_acc, deg_dmg_ge_me_KNN_auc, deg_dmg_ge_me_NB_acc, deg_dmg_ge_me_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_meth, test_xy_gxpr_meth, k)

			print("\nExperiment 1~2. mi, t-SNE + ML")
			num_of_dim_gxpr = train_xy_gxpr.shape[1] - 2
			num_of_dim_meth = train_xy_meth.shape[1] - 2
			num_of_dim_gexp_meth = train_xy_gxpr_meth.shape[1] - 2

			print("num_of_dim_gxpr: " + str(num_of_dim_gxpr))
			print("num_of_dim_meth: " + str(num_of_dim_meth))
			print("num_of_dim_gexp_meth: " + str(num_of_dim_gexp_meth))


			# ExpResult 1. mi, tSNE + ML algorithms with same # of reduced features for each GE, Meth
			# ExpResult 2. mi, tSNE + ML algorithms with same # of reduced features by integrating GE, Meth
			# dimension reduction by t-SNE, mi should be done using training dataset
			# 1.1 t-SNE gexpr, meth
			train_xy_gxpr_tsne = applyDimReduction_TSNE(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/tsne_scatter_plot_gxpr", "train")
			test_xy_gxpr_tsne = applyDimReduction_TSNE(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/tsne_scatter_plot_gxpr", "test")
			train_xy_meth_tsne = applyDimReduction_TSNE(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/tsne_scatter_plot_meth", "train")
			test_xy_meth_tsne = applyDimReduction_TSNE(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/tsne_scatter_plot_meth", "test")

			num_tsne_ge = train_xy_gxpr_tsne.shape[1] - 2
			num_tsne_me = train_xy_meth_tsne.shape[1] - 2

			train_xy_gxpr_meth_tsne = buildIntegratedDataset_DNN(train_xy_gxpr_tsne, train_xy_meth_tsne, mode)
			test_xy_gxpr_meth_tsne = buildIntegratedDataset_DNN(test_xy_gxpr_tsne, test_xy_meth_tsne, mode)

			tsne_ge_RF_acc, tsne_ge_RF_auc, tsne_ge_KNN_acc, tsne_ge_KNN_auc, tsne_ge_NB_acc, tsne_ge_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_tsne, test_xy_gxpr_tsne, k)
			tsne_me_RF_acc, tsne_me_RF_auc, tsne_me_KNN_acc, tsne_me_KNN_auc, tsne_me_NB_acc, tsne_me_NB_auc = doMachineLearning_single_Kfold(train_xy_meth_tsne, test_xy_meth_tsne, k)
			tsne_ge_me_RF_acc, tsne_ge_me_RF_auc, tsne_ge_me_KNN_acc, tsne_ge_me_KNN_auc, tsne_ge_me_NB_acc, tsne_ge_me_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_meth_tsne, test_xy_gxpr_meth_tsne, k)

			# 1.2 mi gexpr, meth
			train_xy_gxpr_mi = applyDimReduction_MI(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/mi_scatter_plot_gxpr", "train")
			test_xy_gxpr_mi = applyDimReduction_MI(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/mi_scatter_plot_gxpr", "test")
			test_xy_meth_mi = applyDimReduction_MI(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/mi_scatter_plot_meth", "test")
			train_xy_meth_mi = applyDimReduction_MI(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv",num_of_dim_meth, dirPath_table1_me + "/mi_scatter_plot_meth", "train")
			# print("n_comp_mi: " + str(n_comp_mi))

			num_mi_ge = train_xy_gxpr_mi.shape[1] - 2
			num_mi_me = train_xy_meth_mi.shape[1] - 2

			train_xy_gxpr_meth_mi = buildIntegratedDataset_DNN(train_xy_gxpr_mi, train_xy_meth_mi, mode)
			test_xy_gxpr_meth_mi = buildIntegratedDataset_DNN(test_xy_gxpr_mi, test_xy_meth_mi, mode)

			mi_ge_RF_acc, mi_ge_RF_auc, mi_ge_KNN_acc, mi_ge_KNN_auc, mi_ge_NB_acc, mi_ge_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_mi, test_xy_gxpr_mi, k)
			mi_me_RF_acc, mi_me_RF_auc, mi_me_KNN_acc, mi_me_KNN_auc, mi_me_NB_acc, mi_me_NB_auc = doMachineLearning_single_Kfold(train_xy_meth_mi, test_xy_meth_mi, k)
			mi_ge_me_RF_acc, mi_ge_me_RF_auc, mi_ge_me_KNN_acc, mi_ge_me_KNN_auc, mi_ge_me_NB_acc, mi_ge_me_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_meth_mi, test_xy_gxpr_meth_mi, k)

			# save the results
			num_deg_ge_List.append(num_deg_ge)
			num_dmg_me_List.append(num_dmg_me)

			num_tsne_ge_List.append(num_tsne_ge)
			num_tsne_me_List.append(num_tsne_me)

			num_mi_ge_List.append(num_mi_ge)
			num_mi_me_List.append(num_mi_me)

			# results
			deg_ge_RF_acc_List.append(deg_ge_RF_acc)
			deg_ge_RF_auc_List.append(deg_ge_RF_auc)
			dmg_me_RF_acc_List.append(dmg_me_RF_acc)
			dmg_me_RF_auc_List.append(dmg_me_RF_auc)
			deg_dmg_ge_me_RF_acc_List.append(deg_dmg_ge_me_RF_acc)
			deg_dmg_ge_me_RF_auc_List.append(deg_dmg_ge_me_RF_auc)

			deg_ge_KNN_acc_List.append(deg_ge_KNN_acc)
			deg_ge_KNN_auc_List.append(deg_ge_KNN_auc)
			dmg_me_KNN_acc_List.append(dmg_me_KNN_acc)
			dmg_me_KNN_auc_List.append(dmg_me_KNN_auc)
			deg_dmg_ge_me_KNN_acc_List.append(deg_dmg_ge_me_KNN_acc)
			deg_dmg_ge_me_KNN_auc_List.append(deg_dmg_ge_me_KNN_auc)

			deg_ge_NB_acc_List.append(deg_ge_NB_acc)
			deg_ge_NB_auc_List.append(deg_ge_NB_auc)
			dmg_me_NB_acc_List.append(dmg_me_NB_acc)
			dmg_me_NB_auc_List.append(dmg_me_NB_auc)
			deg_dmg_ge_me_NB_acc_List.append(deg_dmg_ge_me_NB_acc)
			deg_dmg_ge_me_NB_auc_List.append(deg_dmg_ge_me_NB_auc)

			tsne_ge_RF_acc_List.append(tsne_ge_RF_acc)
			tsne_ge_RF_auc_List.append(tsne_ge_RF_auc)
			tsne_me_RF_acc_List.append(tsne_me_RF_acc)
			tsne_me_RF_auc_List.append(tsne_me_RF_auc)
			tsne_ge_me_RF_acc_List.append(tsne_ge_me_RF_acc)
			tsne_ge_me_RF_auc_List.append(tsne_ge_me_RF_auc)

			tsne_ge_KNN_acc_List.append(tsne_ge_KNN_acc)
			tsne_ge_KNN_auc_List.append(tsne_ge_KNN_auc)
			tsne_me_KNN_acc_List.append(tsne_me_KNN_acc)
			tsne_me_KNN_auc_List.append(tsne_me_KNN_auc)
			tsne_ge_me_KNN_acc_List.append(tsne_ge_me_KNN_acc)
			tsne_ge_me_KNN_auc_List.append(tsne_ge_me_KNN_auc)

			tsne_ge_NB_acc_List.append(tsne_ge_NB_acc)
			tsne_ge_NB_auc_List.append(tsne_ge_NB_auc)
			tsne_me_NB_acc_List.append(tsne_me_NB_acc)
			tsne_me_NB_auc_List.append(tsne_me_NB_auc)
			tsne_ge_me_NB_acc_List.append(tsne_ge_me_NB_acc)
			tsne_ge_me_NB_auc_List.append(tsne_ge_me_NB_auc)

			mi_ge_RF_acc_List.append(mi_ge_RF_acc)
			mi_ge_RF_auc_List.append(mi_ge_RF_auc)
			mi_me_RF_acc_List.append(mi_me_RF_acc)
			mi_me_RF_auc_List.append(mi_me_RF_auc)
			mi_ge_me_RF_acc_List.append(mi_ge_me_RF_acc)
			mi_ge_me_RF_auc_List.append(mi_ge_me_RF_auc)

			mi_ge_KNN_acc_List.append(mi_ge_KNN_acc)
			mi_ge_KNN_auc_List.append(mi_ge_KNN_auc)
			mi_me_KNN_acc_List.append(mi_me_KNN_acc)
			mi_me_KNN_auc_List.append(mi_me_KNN_auc)
			mi_ge_me_KNN_acc_List.append(mi_ge_me_KNN_acc)
			mi_ge_me_KNN_auc_List.append(mi_ge_me_KNN_auc)

			mi_ge_NB_acc_List.append(mi_ge_NB_acc)
			mi_ge_NB_auc_List.append(mi_ge_NB_auc)
			mi_me_NB_acc_List.append(mi_me_NB_acc)
			mi_me_NB_auc_List.append(mi_me_NB_auc)
			mi_ge_me_NB_acc_List.append(mi_ge_me_NB_acc)
			mi_ge_me_NB_auc_List.append(mi_ge_me_NB_auc)


		res_dict = {}
		res_dict["num_deg_ge"] = num_deg_ge_List
		res_dict["num_dmg_me"] = num_dmg_me_List
		res_dict["num_tsne_ge"] = num_tsne_ge_List
		res_dict["num_tsne_me"] = num_tsne_me_List
		res_dict["num_mi_ge"] = num_mi_ge_List
		res_dict["num_mi_me"] = num_mi_me_List

		res_dict["deg_ge_RF_acc"] = deg_ge_RF_acc_List
		res_dict["deg_ge_RF_auc"] = deg_ge_RF_auc_List
		res_dict["dmg_me_RF_acc"] = dmg_me_RF_acc_List
		res_dict["dmg_me_RF_auc"] = dmg_me_RF_auc_List
		res_dict["deg_dmg_ge_me_RF_acc"] = deg_dmg_ge_me_RF_acc_List
		res_dict["deg_dmg_ge_me_RF_auc"] = deg_dmg_ge_me_RF_auc_List

		res_dict["deg_ge_KNN_acc"] = deg_ge_KNN_acc_List
		res_dict["deg_ge_KNN_auc"] = deg_ge_KNN_auc_List
		res_dict["dmg_me_KNN_acc"] = dmg_me_KNN_acc_List
		res_dict["dmg_me_KNN_auc"] = dmg_me_KNN_auc_List
		res_dict["deg_dmg_ge_me_KNN_acc"] = deg_dmg_ge_me_KNN_acc_List
		res_dict["deg_dmg_ge_me_KNN_auc"] = deg_dmg_ge_me_KNN_auc_List

		res_dict["deg_ge_NB_acc"] = deg_ge_NB_acc_List
		res_dict["deg_ge_NB_auc"] = deg_ge_NB_auc_List
		res_dict["dmg_me_NB_acc"] = dmg_me_NB_acc_List
		res_dict["dmg_me_NB_auc"] = dmg_me_NB_auc_List
		res_dict["deg_dmg_ge_me_NB_acc"] = deg_dmg_ge_me_NB_acc_List
		res_dict["deg_dmg_ge_me_NB_auc"] = deg_dmg_ge_me_NB_auc_List

		res_dict["tsne_ge_RF_acc"] = tsne_ge_RF_acc_List
		res_dict["tsne_ge_RF_auc"] = tsne_ge_RF_auc_List
		res_dict["tsne_me_RF_acc"] = tsne_me_RF_acc_List
		res_dict["tsne_me_RF_auc"] = tsne_me_RF_auc_List
		res_dict["tsne_ge_me_RF_acc"] = tsne_ge_me_RF_acc_List
		res_dict["tsne_ge_me_RF_auc"] = tsne_ge_me_RF_auc_List

		res_dict["tsne_ge_KNN_acc"] = tsne_ge_KNN_acc_List
		res_dict["tsne_ge_KNN_auc"] = tsne_ge_KNN_auc_List
		res_dict["tsne_me_KNN_acc"] = tsne_me_KNN_acc_List
		res_dict["tsne_me_KNN_auc"] = tsne_me_KNN_auc_List
		res_dict["tsne_ge_me_KNN_acc"] = tsne_ge_me_KNN_acc_List
		res_dict["tsne_ge_me_KNN_auc"] = tsne_ge_me_KNN_auc_List

		res_dict["tsne_ge_NB_acc"] = tsne_ge_NB_acc_List
		res_dict["tsne_ge_NB_auc"] = tsne_ge_NB_auc_List
		res_dict["tsne_me_NB_acc"] = tsne_me_NB_acc_List
		res_dict["tsne_me_NB_auc"] = tsne_me_NB_auc_List
		res_dict["tsne_ge_me_NB_acc"] = tsne_ge_me_NB_acc_List
		res_dict["tsne_ge_me_NB_auc"] = tsne_ge_me_NB_auc_List

		res_dict["mi_ge_RF_acc"] = mi_ge_RF_acc_List
		res_dict["mi_ge_RF_auc"] = mi_ge_RF_auc_List
		res_dict["mi_me_RF_acc"] = mi_me_RF_acc_List
		res_dict["mi_me_RF_auc"] = mi_me_RF_auc_List
		res_dict["mi_ge_me_RF_acc"] = mi_ge_me_RF_acc_List
		res_dict["mi_ge_me_RF_auc"] = mi_ge_me_RF_auc_List

		res_dict["mi_ge_KNN_acc"] = mi_ge_KNN_acc_List
		res_dict["mi_ge_KNN_auc"] = mi_ge_KNN_auc_List
		res_dict["mi_me_KNN_acc"] = mi_me_KNN_acc_List
		res_dict["mi_me_KNN_auc"] = mi_me_KNN_auc_List
		res_dict["mi_ge_me_KNN_acc"] = mi_ge_me_KNN_acc_List
		res_dict["mi_ge_me_KNN_auc"] = mi_ge_me_KNN_auc_List

		res_dict["mi_ge_NB_acc"] = mi_ge_NB_acc_List
		res_dict["mi_ge_NB_auc"] = mi_ge_NB_auc_List
		res_dict["mi_me_NB_acc"] = mi_me_NB_acc_List
		res_dict["mi_me_NB_auc"] = mi_me_NB_auc_List
		res_dict["mi_ge_me_NB_acc"] = mi_ge_me_NB_acc_List
		res_dict["mi_ge_me_NB_auc"] = mi_ge_me_NB_auc_List

		all_df = pd.DataFrame(res_dict)

		columns_acc = ["num_deg_ge", "deg_ge_RF_acc", "deg_ge_KNN_acc", "deg_ge_NB_acc",
					   "num_dmg_me", "dmg_me_RF_acc", "dmg_me_KNN_acc", "dmg_me_NB_acc",
					   "deg_dmg_ge_me_RF_acc", "deg_dmg_ge_me_KNN_acc", "deg_dmg_ge_me_NB_acc",

					   "num_tsne_ge", "tsne_ge_RF_acc", "tsne_ge_KNN_acc", "tsne_ge_NB_acc",
					   "num_tsne_me", "tsne_me_RF_acc", "tsne_me_KNN_acc", "tsne_me_NB_acc",
					   "tsne_ge_me_RF_acc", "tsne_ge_me_KNN_acc", "tsne_ge_me_NB_acc",

					   "num_mi_ge", "mi_ge_RF_acc", "mi_ge_KNN_acc", "mi_ge_NB_acc",
					   "num_mi_me", "mi_me_RF_acc", "mi_me_KNN_acc", "mi_me_NB_acc",
					   "mi_ge_me_RF_acc", "mi_ge_me_KNN_acc", "mi_ge_me_NB_acc"]
		acc_df = all_df[columns_acc]
		print("acc_df: " + str(acc_df.shape))

		columns_auc = ["num_deg_ge", "deg_ge_RF_auc", "deg_ge_KNN_auc", "deg_ge_NB_auc",
					   "num_dmg_me", "dmg_me_RF_auc", "dmg_me_KNN_auc", "dmg_me_NB_auc",
					   "deg_dmg_ge_me_RF_auc", "deg_dmg_ge_me_KNN_auc", "deg_dmg_ge_me_NB_auc",

					   "num_tsne_ge", "tsne_ge_RF_auc", "tsne_ge_KNN_auc", "tsne_ge_NB_auc",
					   "num_tsne_me", "tsne_me_RF_auc", "tsne_me_KNN_auc", "tsne_me_NB_auc",
					   "tsne_ge_me_RF_auc", "tsne_ge_me_KNN_auc", "tsne_ge_me_NB_auc",

					   "num_mi_ge", "mi_ge_RF_auc", "mi_ge_KNN_auc", "mi_ge_NB_auc",
					   "num_mi_me", "mi_me_RF_auc", "mi_me_KNN_auc", "mi_me_NB_auc",
					   "mi_ge_me_RF_auc", "mi_ge_me_KNN_auc", "mi_ge_me_NB_auc"]
		auc_df = all_df[columns_auc]
		print("auc_df: " + str(auc_df.shape))

		acc_df.to_csv(output_dir + "/["+ mode + "] ML acc_k1-5.tsv", header=True, index=False, sep="\t")
		auc_df.to_csv(output_dir + "/["+ mode + "] ML auc_k1-5.tsv", header=True, index=False, sep="\t")


## main
if __name__ == '__main__':
	help_str = "python AD_Prediction_ML.py" + "\n"

	# input directory
	input_dir_path = r"......\results\k_fold_train_test"

	# output directory
	output_dir_path = r"......\results\k_fold_train_test_results"
	if not os.path.exists(output_dir_path): os.mkdir(output_dir_path)

	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=str, default=input_dir_path, help=help_str)
	parser.add_argument("--output", type=str, default=output_dir_path, help=help_str)

	args = parser.parse_args()
	main(args)
