import csv
import time
import asyncio
from openai import OpenAI
from string import Template




# Load inputs and create asynchronous iterator
input_file = "/pbs/throng/training/universite-hiver/wschooloptim/resinwinterschool/tweets_spanish_english.csv"
content_column = 'english'

start = time.time()
with open(input_file, 'r') as f:
    inputs = [d[content_column] for d in csv.DictReader(f)]
end = time.time()
print(f"Loading {len(inputs)} tweets tooks {end - start} seconds.")

async def inputsIterator(start, end):
    for input_ in inputs[start:end]:
        yield input_

# Load choices
choices_file = "/pbs/throng/training/universite-hiver/wschooloptim/resinwinterschool/multiple.txt"
with open(choices_file, 'r') as f:
    choice = f.read().split("\n")


extra_body = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 16,
    "repetition_penalty": 1.2,
    "seed": 19,
    "max_tokens": 256
}
# instructions = "Please translate the following text from spanish to english. Do not provide any explanation or note. Be consice and do not produce anything other than a literal translation."
instructions = "Please classify the following social media message (posted in the weeks leading up to the 2025 Chilean presidential election) according to whether it explicitly expresses the intention  of the author to vote for or calls for a vote for any of the candidates in that election: Jeannette Jara, José Antonio Kast, Johannes Kaiser, Evelyn Matthei, Franco Parisi, Eduardo Artés, Harold Mayne-Nichols, Marco Enríquez-Ominami (also known as MEO), or whether it expresses neither of these intentions. Your answer should be based solely on the information contained in the message. Do not confuse a simple mention of a candidate with an intention or call to vote for him. Do not assume that a message containing only positive opinions about a particular candidate explicitly expresses the intention to vote for that candidate. Do not assume that a message corresponding to the opinions or political positions of a particular candidate necessarily expresses the intention to vote for that candidate. Do not confuse retweets, indirect speech or a quote to another person with the opinion of the author of the message. Be concise and respond only with the last name or the word 'None'. Here is the message: ${tweet}"

# vllm server
# export=CUDA_VISIBLES_DEVICES=1; vllm serve HuggingFaceH4/zephyr-7b-beta --max_model_len=10000 --guided_decoding_backend=xgrammar --seed=1  --tensor-parallel-size 1 &
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# ollama server
# OLLAMA_MODELS=/pbs/throng/training/universite-hiver/wschooloptim/cache/ollama/models; CUDA_VISIBLES_DEVICES=GPU-99ec877c-53b2-49ba-a1ea-0b6f0d0f0c08; OLLAMA_HOST=http://127.0.0.1:11435; OLLAMA_FLASH_ATTENTION=1; ollama serve &
ollama_client = OpenAI(
    base_url="http://127.0.0.1:11435/v1/",
    api_key="EMPTY"
)


async def doCompletetion(input_):
    # make request

    # res = ollama_client.responses.create(
    #     model=ollama_client.models.list().data[0].id,
    #     instructions=instructions,
    #     input=input_[content_column],
    #     extra_body=extra_body)
    # format result
    # input_.update({'res': res.output_text.strip()})
    # and return
    # return res

    return vllm_client.chat.completions.create(
        model=vllm_client.models.list().data[0].id,
        messages=[{
            'role': 'user',
            'content': Template(instructions).substitute(tweet=input_)
        }],
        extra_body={"structured_outputs": {"choice": choice}},
        max_completion_tokens=extra_body["max_tokens"],
        temperature=extra_body["temperature"],
        top_p=extra_body["top_p"],
    )

async def run_all(start, end):
    # Asynchronously call the function for each prompt
    tasks = [
        doCompletetion(input_)
        async for input_ in inputsIterator(start, end)
    ]
    # Gather and run the tasks concurrently
    results = await asyncio.gather(*tasks)
    return results

# Run all courutines
init_idx = 115789
nb_tweets = 50
start = time.time()
results = asyncio.run(run_all(start=init_idx, end=init_idx + nb_tweets))
elapsed = time.time() - start
print(f"""
Annotationg {nb_tweets} tweets took {elapsed} seconds,
this is {elapsed / nb_tweets} seconds per tweet or
{(1 / (24 * 3600)) * len(inputs) * elapsed / nb_tweets} days for the whole database of {len(inputs)} inputs.
""")


# vllm
# Annotationg 50 tweets took 2.6444010734558105 seconds,
# this is 0.05288802146911621 seconds per tweet or
# 0.6389858522573003 days for the whole database of 1043873 inputs.

# ollama
# Annotationg 50 tweets took 26.955878496170044 seconds,
# this is 0.5391175699234009 seconds per tweet or
# 6.5135448503316 days for the whole database of 1043873 inputs.
