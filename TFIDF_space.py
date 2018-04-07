#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# 引入Bunch类  
from sklearn.datasets.base import Bunch
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取文件  
def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

# 读取bunch对象 
def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

# 写入bunch对象 
def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

#该函数用于创建TF-IDF词向量空间  
def vector_space(stopword_path,bunch_path,space_path,train_tfidf_path=None):

    stpwrdlst = _readfile(stopword_path).splitlines() # 读取停用词  
    bunch = _readbunchobj(bunch_path) # 读取Bunch对象
    # 构建tf-idf词向量空间对象  
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})
    # tdm是权值矩阵
    # vocabulary是词典索引, 即每个词对应的序号

    if train_tfidf_path is not None:
        trainbunch = _readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5) # 使用TfidfVectorizer初始化向量空间模型
        # sublinear_tf=true: 计算tf值采用亚线性策略, 
        # 比如, 我们以前算tf是词频, 现在用1+log(tf)来充当词频
        # max_df=0.5: 把出现频率超过50%的词设为停用词, 比如一个词"XX"在tmt中的50%以上的文档都出现了,
        # 那么它就没有很强的分类价值, 可以将其设为停用词.
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents) # tdm中存的就是权值矩阵
        tfidfspace.vocabulary = vectorizer.vocabulary_

    _writebunchobj(space_path, tfidfspace)

if __name__ == '__main__':

	stopword_path = "train_word_bag/hlt_stop_words.txt"
	bunch_path = "train_word_bag/train_set.dat"
	space_path = "train_word_bag/tfdifspace.dat"
	vector_space(stopword_path,bunch_path,space_path)
	print "train TF-IDF created"

	bunch_path = "test_word_bag/test_set.dat"
	space_path = "test_word_bag/testspace.dat"
	train_tfidf_path="train_word_bag/tfdifspace.dat"
	vector_space(stopword_path,bunch_path,space_path,train_tfidf_path)
	print "test TF-IDF created"

