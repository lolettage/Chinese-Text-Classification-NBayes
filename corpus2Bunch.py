#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os # python内置的包, 用于进行文件目录操作, 之后将会用到os.listdir函数
import cPickle as pickle # 导入cPickle包并且取一个别名pickle

from sklearn.datasets.base import Bunch

def _readfile(path):
    '''读取文件'''
    with open(path, "rb") as fp:
        content = fp.read()
    return content

def corpus2Bunch(wordbag_path,seg_path):
    catelist = os.listdir(seg_path) # 获取seg_path下的所有子目录, 也就是分类信息
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    '''
    extend(addlist)是python list中的函数, 
    意思是用新的list(addlist)去扩充原来的list
    '''

    # 获取每个目录下所有的文件
    for mydir in catelist:
        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:  # 遍历类别目录下文件
            fullname = class_path + file_path  # 拼出文件名全路径
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(_readfile(fullname))  # 读取文件内容

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)

if __name__ == "__main__":
    # 对训练集进行Bunch化操作：
	wordbag_path = "train_word_bag/train_set.dat"  # Bunch存储路径
	seg_path = "train_corpus_seg/" 
	corpus2Bunch(wordbag_path, seg_path)
	print "train Bunch created"

    # 对测试集进行Bunch化操作：
	wordbag_path = "test_word_bag/test_set.dat"  # Bunch存储路径
	seg_path = "test_corpus_seg/"  
	corpus2Bunch(wordbag_path, seg_path)
	print "test Bunch created"




