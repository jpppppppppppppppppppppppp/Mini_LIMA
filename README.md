## Self Instruction

All generated instructions are in folder [tasks](./tasks). All generated instances are in folder [instance](./instance).

You can run [instruction_gen.py](./code/instruction_gen.py) to generate instructions, [instruction_cls_gen.py](./code/instruction_cls_gen.py) to generate classification tasks. Some hyper-parameters you can modify:

- `num_instructions_to_generate`
- `machine_sample` and `seed_sample` : the numbers of tasks to format the prompt.
- `machine_gen`: the number of tasks to generate in each request.

Run [instruction_classify.py](./code/instruction_classify.py) to prompt the LLM to classify each tasks into classification tasks or non-classification tasks.

Run [cls.py](./code/cls.py) or [none_cls.py](./code/none_cls.py) to generate instance with different approaches.

You can use [total.py](./code/total.py) to sum all above works up.

## Fine-tune

The [run.sh](./finetune/run.sh) is enough to make fine-tune work. The dataset are saved in folder [dataset](./dataset).

## Evaluate

```shell
export OPENAI_API_BASE=<YOUR_OPENAI_API_BASE>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
export HF_ENDPOINT=https://hf-mirror.com

git clone https://github.com/tatsu-lab/alpaca_eval.git
cd alpaca_eval/ &&pip install -e . && cd ..

alpaca_eval evaluate_from_model \
  --model_configs 'mini_lima' \
  --annotators_config 'chatgpt'
```

