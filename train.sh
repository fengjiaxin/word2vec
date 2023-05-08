./word2vec skipgram -input file/enwik9_100000.txt -output result/file9 -lr 0.025 -dim 32 \
  -ws 5 -epoch 5 -minCount 5 -neg 5 -loss ns \
  -thread 12 -lrUpdateRate 100
