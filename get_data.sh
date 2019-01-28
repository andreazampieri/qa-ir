mkdir qa-ir/data
mkdir qa-ir/data/raw
cd qa-ir/original_data && wget http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2 && tar xvjf jacana-qa-naacl2013-data-results.tar.bz2 && unzip WikiQACorpus.zip && cd ..;
pwd
mv original_data/jacana-qa-naacl2013-data-results data/raw/trecqa
mv original_data/WikiQACorpus data/raw/wikiqa
pwd
gzip -d ../data/raw/trecqa/train2393.cleanup.xml.gz
