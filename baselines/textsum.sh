#!/usr/bin/env bash

pushd
cd ../existing_models/textsum
python data_convert_example.py \
  --command text_to_binary \
  --in_file ../../data/nips-papers/paper_titles_and_abstracts.csv \
  --out_file data/binary_data

popd