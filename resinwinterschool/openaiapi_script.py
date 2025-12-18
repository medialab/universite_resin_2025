from openai import OpenAI

# export=CUDA_VISIBLES_DEVICES=1; vllm serve HuggingFaceH4/zephyr-7b-beta --max_model_len=10000 --guided_decoding_backend=xgrammar --seed=1  --tensor-parallel-size 1 &

instructions = "Please translate the following text from spanish to english. Do not provide any explanation or note. Be consice and do not produce anything other than a literal translation."
input = "Me gusta comer papas."

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': f"{instructions}; '{input}'",
        }
    ],
    model=v_client.models.list().data[0].id,
)
print(chat_completion.choices[0].message.content)

# -------------------------------------------------

# export=CUDA_VISIBLES_DEVICES=0; ollama serve &

o_client = OpenAI(
    base_url="http://127.0.0.1:11435/v1/",
    api_key="ollama"
)

chat_completion = o_client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': f"{instructions}; '{input}'",
        }
    ],
    model=o_client.models.list().data[0].id,
)
print(chat_completion.choices[0].message.content)
