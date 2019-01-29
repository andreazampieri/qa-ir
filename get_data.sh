mkdir data
mkdir data/raw
cd original_data && wget http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2 && tar xvjf jacana-qa-naacl2013-data-results.tar.bz2 && unzip WikiQACorpus.zip && cd ..;
mkdir data/raw/trecqa
mv original_data/jacana-qa-naacl2013-data-results data/raw/trecqa
mkdir data/raw/wikiqa
mv original_data/WikiQA-dev.tsv data/raw/wikiqa/
mv original_data/WikiQA-train.tsv data/raw/wikiqa/
mv original_data/WikiQA-test.tsv data/raw/wikiqa/
gzip -d data/raw/trecqa/train2393.cleanup.xml.gz
