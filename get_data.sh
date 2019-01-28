mkdir qa-ir/data
mkdir qa-ir/data/raw
cd qa-ir/original_data && wget http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2 && tar xvjf jacana-qa-naacl2013-data-results.tar.bz2 && unzip WikiQACorpus.zip

mv qa-ir/original_data/jacana-qa-naacl2013-data-results qa-ir/data/raw/trecqa
mv qa-ir/original_data/WikiQACorpus qa-ir/data/raw/wikiqa

gzip -d qa-ir/data/raw/trecqa/train2393.cleanup.xml.gz
