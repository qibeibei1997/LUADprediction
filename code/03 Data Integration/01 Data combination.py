import pandas as pd
import numpy as np
import random as rd
import csv


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

	for idx_g in range(0, n_row_g ):
		label_g = xy_gxpr[idx_g][-2:]
		#print(label_g)
		for idx_m in range(0, n_row_m ):
			label_m = xy_meth[idx_m][-2:]
			# print(label_m)

			# normal
			if np.array_equal(label_g, NoArr) and np.array_equal(label_m, NoArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_No.add(integ_idx)
				# print(integ_idx)
				# print("normal: " + integ_idx)
				NoCnt += 1

			# Tu
			if np.array_equal(label_g, TuArr) and np.array_equal(label_m, TuArr):
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

		# for Tu
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

		# for Tu
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


	xy_me_ge_values = np.array(xy_gxpr_meth)
	print(xy_me_ge_values.shape)

	return xy_me_ge_values


def buildIntegratedDataset_DNN_selectN(xy_gxpr, xy_meth, mode, nSamples):

	print("buildIntegratedDataset2")
	xy_gxpr_meth = []

	n_row_g, n_col_g = xy_gxpr.shape
	n_row_m, n_col_m = xy_meth.shape

	# build random index pair set
	idxSet_No = set()
	idxSet_Tu = set()

	NoArr = [1,0]
	TuArr = [0,1]
	NoCnt = 0
	TuCnt = 0

	for idx_g in range(0, n_row_g):
		label_g = xy_gxpr[idx_g][-2:]

		for idx_m in range(0, n_row_m):
			label_m = xy_meth[idx_m][-2:]

			# normal
			if np.array_equal(label_g, NoArr) and np.array_equal(label_m, NoArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_No.add(integ_idx)
				NoCnt += 1

			# Tu
			if np.array_equal(label_g, TuArr) and np.array_equal(label_m, TuArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_Tu.add(integ_idx)
				TuCnt += 1

	print("NoCnt: " + NoCnt.__str__())
	print("TuCnt: " + TuCnt.__str__())

	balanced_sample_size = 0;
	if (NoCnt > TuCnt):
		balanced_sample_size = TuCnt

	if (NoCnt < TuCnt):
		balanced_sample_size = NoCnt

	idxList_No = list(idxSet_No)
	idxList_Tu = list(idxSet_Tu)
	rd.shuffle(idxList_No)  ## randomly suffle the samples
	rd.shuffle(idxList_Tu)  ## randomly suffle the samples

	print("idxList_No: " + str(idxList_No))
	print("idxList_Tu: " + str(idxList_Tu))

	if mode == "balanced":
		print("balanced mode")

		# for normal
		cnt = 0
		for idx in idxList_No:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if (cnt >= balanced_sample_size) or (cnt >= nSamples):
				break

		# for Tu
		cnt = 0
		for idx in idxList_Tu:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if (cnt >= balanced_sample_size) or (cnt >= nSamples):
				break

	if mode != "balanced":
		print("unbalanced mode")

		# for normal
		for idx in idxList_No:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

		# for Tu
		for idx in idxList_Tu:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

	xy_me_ge_values = np.array(xy_gxpr_meth)
	print(xy_me_ge_values.shape)

	return xy_me_ge_values


XY_gxpr = pd.read_csv(r"........\gene.csv", encoding="UTF-8", low_memory=False, index_col=0)
print(XY_gxpr)
XY_gxpr = XY_gxpr.values
print(XY_gxpr)
print(XY_gxpr.shape)
XY_meth = pd.read_csv(r"........\meth.csv", encoding="UTF-8", low_memory=False, index_col=0)
print(XY_meth)
XY_meth = XY_meth.values
print(XY_meth)
print(XY_meth.shape)

input_data_mode = "all"
if input_data_mode == "all":
	# integrate two heterogenous dataset
	XY_gxpr_meth = buildIntegratedDataset_DNN(XY_gxpr, XY_meth, "unbalanced")
else:
	# integrate two heterogenous dataset with randomly selected N samples 
	nSamples = 2000  # for each label
	XY_gxpr_meth = buildIntegratedDataset_DNN_selectN(XY_gxpr, XY_meth, "unbalanced", nSamples)

print("final XY_gxpr_meth: " + str(XY_gxpr_meth.shape)) ## row: 2000 samples col: 193 (191 features + 2 labels)
print(XY_gxpr_meth)

with open(r"XY_gxpr_meth.csv", 'w+', encoding='utf-8', newline='') as file:
	writer = csv.writer(file)
	writer.writerows(XY_gxpr_meth)
