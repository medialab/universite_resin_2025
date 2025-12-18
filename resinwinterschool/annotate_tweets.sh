python annotate_tweets.py \
       --model_params='{"model": "HuggingFaceH4/zephyr-7b-beta", "max_model_len": 10000, "guided_decoding_backend": "xgrammar", "seed": 1, "tensor_parallel_size": 1}' \
       --sampling_params='{"temperature": 0.7, "top_p": 0.95, "top_k": 50, "max_tokens": 16, "repetition_penalty": 1.2, "seed": 19, "max_tokens": 256}' \
       --tweets_file=tweets_spanish_english_25K.csv \
       --tweets_column=english \
       --system_prompt=system_prompt_english.txt \
       --user_prompt=user_prompt_voteintention_multiple_all_english.txt \
       --guided_choice=multiple.txt \
       --batch_size=5000 \
       --outfolder=tmp


