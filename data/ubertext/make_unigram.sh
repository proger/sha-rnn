spm_train --input=sentences10m.txt --model_prefix=unigram --vocab_size=32768 --character_coverage=0.9995 --model_type=unigram --num_threads=32 --shuffle_input_sentence=true --normalization_rule_name=nfkc
spm_export_vocab --model=unigram.model --output_format=syms > unigram.syms
