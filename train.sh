mkdir -p result
./word2vec skipgram -input file/seg_title.txt -output result/title -lr 0.025 -dim 32 \
  -ws 5 -epoch 3 -minCount 3 -neg 5 -loss ns \
  -thread 12 -lrUpdateRate 100
