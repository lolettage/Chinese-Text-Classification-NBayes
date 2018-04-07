#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import jieba
# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 分词结果的保存路径
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)

# 读取待分词文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

def corpus_segment(corpus_path, seg_path):
    '''
    corpus_path: 原始训练数据的路径
    seg_path: 将原始训练数据分词后的路径
    '''
    catelist = os.listdir(corpus_path)  # 获取corpus_path下的所有子目录
    '''
    其中子目录的名字就是类别名，例如：
    train_corpus/tmt/tmt_21.txt中，'train_corpus/'是corpus_path，'tmt'是catelist中的一个成员
    '''

    # 获取每个目录(类别)下所有的文件
    for mydir in catelist:
        '''
        这里mydir就是train_corpus/tmt/21.txt中的tmt(即catelist中的一个类别)
        '''
        class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径如: train_corpus/tmt/
        seg_dir = seg_path + mydir + "/"  # 拼出分词后存贮的对应目录路径如: train_corpus_seg/tmt/

        if not os.path.exists(seg_dir):  # 是否存在分词目录, 如果没有则创建该目录
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)  # 获取未分词语料库中某一类别中的所有文本
        '''
        例如: 
        train_corpus/tmt/中的
        tmt_21.txt,
        tmt_22.txt,
        tmt_23.txt
        ...
        file_list=[..., 'tmt_21.txt', 'tmt_22.txt', ...]
        '''
        for file_path in file_list:  # 遍历类别目录下的所有文件
            fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
            content = readfile(fullname)  # 读取文件内容

            content = content.replace("\r\n", "")  # 删除换行
            content = content.replace(" ", "") # 删除空行、多余的空格
            content_seg = jieba.cut(content)  # 为文件内容分词
            savefile(seg_dir + file_path, " ".join(content_seg))  # 将处理后的文件保存到分词后语料目录

if __name__=="__main__":
    #对训练集进行分词
	corpus_path = "./train_corpus/"  
	seg_path = "./train_corpus_seg/" 
	corpus_segment(corpus_path,seg_path)
	print "train corpus' word segment is over"

    #对测试集进行分词
	corpus_path = "./test_corpus/" 
	seg_path = "./test_corpus_seg/"  
	corpus_segment(corpus_path,seg_path)
	print "test corpus' word segment is over"
