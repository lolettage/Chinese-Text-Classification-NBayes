#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import cPickle as pickle
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法


# 读取bunch对象
def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

# 导入训练集
trainpath = "train_word_bag/tfdifspace.dat" # tfdifspace是一个bunch结构, 包括了tdm矩阵和词典(vocabulary)
train_set = _readbunchobj(trainpath)

# 导入测试集
testpath = "test_word_bag/testspace.dat"
test_set = _readbunchobj(testpath)

# 训练分类器: 输入tdm矩阵(词向量)和分类标签, alpha:0.01 alpha越小, 迭代次数越多, 精度越高
clf = MultinomialNB(alpha=0.01).fit(train_set.tdm, train_set.label)

# 预测分类结果
predicted = clf.predict(test_set.tdm)
tmt_count = 0
food_count = 0
eng_count = 0

for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
		# print file_name,": 实际类别:",flabel," -->预测类别:",expct_cate
		if expct_cate=="tmt":
			tmt_count = tmt_count + 1
		if expct_cate=="food":
			food_count = food_count + 1
		if expct_cate=="eng":
			eng_count = eng_count + 1

total = eng_count + food_count + tmt_count
tmt_ratio = float(tmt_count) / float(total)
food_ratio = float(food_count) / float(total)
eng_ratio = float(eng_count) / float(total)

print "predict result: "
print "tmt_ratio = %r" %(tmt_ratio)
print "food_ratio = %r" %(food_ratio)
print "eng_ratio = %r" %(eng_ratio)



