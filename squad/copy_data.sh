mkdir -p data
cd data
curl -O http://kimi.ml.cmu.edu/structure/squad/data_cache.tar
tar xvf data_cache.tar
curl -O http://kimi.ml.cmu.edu/structure/squad/elmo_ee.pt
pip install allennlp
