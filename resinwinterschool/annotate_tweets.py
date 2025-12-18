import os
import sys
import csv
import json
import time
import sqlite3
import logging
import numpy as np

from string import Template
from argparse import ArgumentParser

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

DEFAULTOUTFOLDER = "tmp"
DEFAULTBATCHSIZE = 5000
DEFAULTMODELPARAMS = '{"model": "mistralai/Mistral-Small-24B-Instruct-2501", "tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral", "guided_decoding_backend": "xgrammar", "seed": 1, "tensor_parallel_size": 1}'
DEFAULTSAMPLEPARAMS = '{"temperature": 0.15, "seed": 1, "max_tokens": 256}'


def writeCsv(file, rows, headers, logger, verbose=True):
    if not isinstance(headers, list):
        raise ValueError(
            f"Headers must be a list. Found {type(headers)} for '{headers}'.")
    with open(file, 'w') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(rows)
    if verbose:
        logger.info(f"Csv file saved at {file}")


def make_logger(logfile):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(logfile, 'w', 'utf-8'),
            logging.StreamHandler(sys.stdout)])
    return logger

def make_prompts(system_prompt, user_prompt, tweets):
    return [
        [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": Template(user_prompt).substitute(tweet=tweet)
            }
        ] for tweet in tweets]


def compute_llm_asnwers(data, model, sampling_params, system_prompt, user_prompt, output_folder, logger):

    logger.info(
        f"Computing llm answers {len(data)} bios.")

    outcolumns = [
        "id",
        "tweet",
        'answer'
    ]

    data_length = len(data)
    batchl = [
        [i * batch_size, (i + 1) * batch_size]
        for i in range(np.int32(data_length / batch_size + 1))
    ]
    batchl[-1][1] = min(data_length,  batchl[-1][1])

    tweets_idx = range(len(data))

    for batch_idx, b in enumerate(batchl):

        if len(tweets) == 0:
            continue

        file = os.path.join(output_folder, f"llm_answer_{batch_idx}.csv")
        lockfile = file + ".lock"

        # If batch already computed continue or lock is granted, continue
        if os.path.exists(file) or os.path.exists(lockfile):
            logger.info(f"Already computed file at {file}. Continuing.")
            continue
        # If not, create lock and start batch computation
        with open(lockfile, 'w') as f:
            csv.writer(f).writerow(outcolumns)
        logger.info(f"Computing batch at index {batch_idx}...")

        prompts = make_prompts(system_prompt, user_prompt, data[b[0]: b[1]])

        start = time.time()
        outputs = model.chat(
            messages=prompts,
            sampling_params=sampling_params,
            use_tqdm=True)
        elapsed = time.time() - start
        logger.info(f"Annotationg {len(prompts)} tweets took {elapsed} seconds, this is {elapsed / len(prompts)} seconds per tweet.")

        parsed_outputs = [o.outputs[0].text for o in outputs]

        rows = zip(tweets_idx[b[0]: b[1]], data[b[0]: b[1]], parsed_outputs)

        # write computed data
        writeCsv(file, rows, outcolumns, logger)
        # and release lock
        os.system(f"rm {lockfile}")


if __name__ == "__main__":

    """
       python annotate_tweets.py \
              --model_params='{"model": "HuggingFaceH4/zephyr-7b-beta", "max_model_len": 10000, "guided_decoding_backend": "xgrammar", "seed": 1, "tensor_parallel_size": 1}' \
              --sampling_params='{"temperature": 0.7, "top_p": 0.95, "top_k": 50, "max_tokens": 16, "repetition_penalty": 1.2, "seed": 19, "max_tokens": 256}' \
              --tweets_file=tweets_spanish_english.csv \
              --tweets_column=english \
              --system_prompt=system_prompt_english.txt \
              --user_prompt=user_prompt_voteintention_multiple_all_english.txt \
              --guided_choice=multiple.txt \
              --batch_size=5000 \
              --outfolder=tmp
    """

    ap = ArgumentParser()
    ap.add_argument('--model_params', type=str, default=DEFAULTMODELPARAMS)
    ap.add_argument('--sampling_params', type=str, default=DEFAULTSAMPLEPARAMS)
    ap.add_argument('--batch_size', type=int, default=DEFAULTBATCHSIZE)

    ap.add_argument('--system_prompt', required=True, type=str)
    ap.add_argument('--user_prompt', required=True, type=str)
    ap.add_argument('--guided_choices', required=False, type=str, default='')

    ap.add_argument('--tweets_file', required=True, type=str)
    ap.add_argument('--tweets_column', required=True, type=str)

    ap.add_argument('--logfile', type=str, default=None)
    ap.add_argument('--outfolder', type=str, default=DEFAULTOUTFOLDER)

    args = ap.parse_args()

    model_params = json.loads(args.model_params)
    sampling_params = json.loads(args.sampling_params)
    batch_size = args.batch_size

    system_prompt = args.system_prompt
    user_prompt = args.user_prompt
    guided_choices = args.guided_choices
    tweets_file = args.tweets_file
    tweets_column = args.tweets_column

    outfolder = args.outfolder
    logfile = args.logfile if args.logfile is not None else os.path.join(outfolder, "out.log")

    os.makedirs(outfolder, exist_ok=True)

    # 0/ Make logger and log parameters
    logger = make_logger(logfile)

    parameters = vars(args)
    dumped_parameters = json.dumps(parameters, sort_keys=True, indent=4)
    logger.info("---------------------------------------------------------")
    logger.info(f"PARAMETERS:\n{dumped_parameters[2:-2]}")

    # 1/ Load prompts and choices
    if os.path.exists(system_prompt):
        logger.info(f"System prompt loaded from file at {system_prompt}")
        with open(system_prompt, 'r') as f:
            system_prompt = f.read()
    logger.info(f"System prompt:\n\t{system_prompt}")

    if os.path.exists(user_prompt):
        logger.info(f"User promt loaded from file at {user_prompt}")
        with open(user_prompt, 'r') as f:
            user_prompt = f.read()
    logger.info(f"User prompt:\n\t{user_prompt}")

    if os.path.exists(guided_choices):
        logger.info(f"Choices loaded from file at {guided_choices}")
        with open(guided_choices, 'r') as f:
            guided_choices = [l[0] for l in csv.reader(f)]
    else:
        guided_choices = guided_choices.split(',')
    logger.info(f"Choices:\n\t{guided_choices}")

    # 2/ Load data (tweets) to be used in prompts
    if not os.path.exists(tweets_file):
        raise ValueError(f"Unnable to find tweets file at {tweets_file}")
    with open(tweets_file, newline='') as f:
        csvFile = csv.DictReader(f)
        tweets = [l[tweets_column] for l in csvFile]
    print(f"Load {len(tweets)} tweets from column {tweets_column} on {tweets_file}.")

    # 3/ Set sampling params
    if guided_choices:
        guided_decoding_params = GuidedDecodingParams(choice=guided_choices)
        sp = SamplingParams(
            **sampling_params,
            guided_decoding=guided_decoding_params)
    else:
        sp = SamplingParams(**sampling_params)
    logger.info(f"LLM model was loaded. Sampling params are {sp}.")

    # 4/ Load model
    model = LLM(**model_params)

    # 5/ Compute answers
    start = time.time()
    compute_llm_asnwers(
        tweets, model, sp,
        system_prompt, user_prompt,
        outfolder, logger)
    elapsed = time.time() - start
    logger.info(f"Annotationg {len(tweets)} tweets took {elapsed} seconds, this is {elapsed / len(tweets)} seconds per tweet.")
