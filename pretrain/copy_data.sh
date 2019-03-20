mkdir -p ../data
cd ../data
wget http://kimi.ml.cmu.edu/structure/wiki.tar
rm -rf wiki
tar xvf wiki.tar
cd ../pretrain
curl -O http://kimi.ml.cmu.edu/structure/test_cachev2.pkl
curl -O http://kimi.ml.cmu.edu/structure/testv2.txt
curl -O http://kimi.ml.cmu.edu/structure/trainv2.txt
curl -O http://kimi.ml.cmu.edu/structure/valid_cachev2.pkl
curl -O http://kimi.ml.cmu.edu/structure/validv2.txt
curl -O http://kimi.ml.cmu.edu/structure/vocabv2.pkl
