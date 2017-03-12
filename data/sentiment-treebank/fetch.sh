#rm stanfordSentimentTreebank.zip
#wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
#unzip stanfordSentimentTreebank.zip
#rm -rf __MACOSX
rm trainDevTestTrees_PTB.zip
wget http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip trainDevTestTrees_PTB.zip
sudo pip install pytreebank
