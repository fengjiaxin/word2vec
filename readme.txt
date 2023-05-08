从fasttext 去掉一些功能，简化代码， 生成word2vec 的cpp项目

项目简述

alias_sample.h : 别名采样，O(1)时间按照频率随机选择
args.h/args.cpp : 参数
dictionary.h/dictionary.cpp : 字典，读取预料，根据词的频率生成词典相关信息
loss.h/loss.cpp : 训练模型的损失函数， negativeSample, 负采样
math_helper.h: 快速计算 log,sigmoid的方法
matrix.h/matrix.cpp : 矩阵，对应 input/output 的矩阵
vector.h/vector.cpp : 向量， 对应梯度向量，隐藏向量等
model.h/model.cpp : 负责更新 input/output向量，计算损失函数等功能
word2vec.h/word2vec.cpp : 功能的集合，读取数据，训练模型，存储模型等
main.cpp : 主文件


1. 构建项目的命令
rm -rf build
mkdir build
cd build
cmake ..
make
cp word2vec ../

2. 训练模型, 训练语料已经放在file中
mkdir -p result
./word2vec skipgram -input file/enwik9_100000.txt -output result/file9 -lr 0.025 -dim 32 \
  -ws 5 -epoch 2 -minCount 5 -neg 5 -loss ns \
  -thread 4 -lrUpdateRate 100

3. 测试模型
3.1 获取word 向量 ./word2vec print-word-vectors result/file9.bin
3.2 找出相似词 ./word2vec nn result/file9.bin
3.3 查看模型信息 ./word2vec dump result/file9.bin args


