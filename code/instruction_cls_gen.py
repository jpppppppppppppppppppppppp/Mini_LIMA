import os
import json
import random
import re
import string
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rouge_score import rouge_scorer
from functools import partial
from openai import OpenAI
import time

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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"
)

random.seed(time.time())
num_instructions_to_generate = 10000
machine_sample = 6
seed_sample = 12
machine_gen = 18


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)

def handle_completion(comp, num):
    if comp.choices == None:
        return []
    if len(comp.choices) == 1:
        print(f"content:\n{comp.choices[0].message.content}")
        if comp.choices[0].finish_reason == "length":
            return []
        cho = comp.choices[0].message.content
        raw = re.findall(r'(\s*\d+\.\s*)?(\s*Task\s*\d+\.\s*)?(.*)', cho)
        ret = []
        for inss in raw:
            for ins in inss:
                ins = re.sub(r"\s+", " ", ins).strip()
                ins = ins.strip().capitalize()
                if ins == "":
                    continue
                if len(ins.split()) <= 3 or len(ins.split()) > 150:
                    continue
                if any(find_word_in_string(word, ins) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to", "music", "powerpoint", "rite a program", "rite a code", "rite a code"]):
                    continue
                if ins.startswith("Write a program"):
                    continue
                if ins.startswith("Write a code"):
                    continue
                if ins.startswith("Write a script"):
                    continue
                if not ins[0].isascii():
                    continue
                if ins[0].isnumeric():
                    continue
                if ins[0] in string.punctuation:
                    continue
                ret.append(ins)
        return ret
    return []

if __name__ == "__main__":
    seed_tasks = [json.loads(line) for line in open(r"D:\Desktop\dsa\seed_tasks.jsonl", "r")]
    seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [item["instruction"] for item in seed_tasks]
    print("Have loaded {} tasks".format(len(seed_tasks)))
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    machine_tasks = []
    machine_req_idx = 0
    idx = int(num_instructions_to_generate / 1000)
    for i in range (1, idx+1):
        if os.path.exists(r"D:\Desktop\dsa\cly_machine_tasks-{}.jsonl".format(i)):
            with open(r"D:\Desktop\dsa\cly_machine_tasks-{}.jsonl".format(i), "r") as fw:
                for line in fw:
                    ins = json.loads(line)
                    machine_tasks.append(ins)
                    machine_req_idx = ins["machine_req_idx"] + 1
            print("Have loaded {} tasks".format(len(machine_tasks)))
    machine_instructions = [item["instruction"] for item in machine_tasks]
    while len(machine_instructions) < num_instructions_to_generate:
        with open(r"D:\Desktop\dsa\cly_machine_tasks-{}.jsonl".format(int((len(machine_instructions) + 999)/1000 + ((len(machine_instructions) % 1000) == 0))), "a") as fw:
            prompts = []
            round = 1
            for tt in range(round):
                prompt_instructions = random.sample(machine_instructions, min(machine_sample, len(machine_instructions)))
                prompt_instructions += random.sample(seed_instructions, machine_sample + seed_sample - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                prompt = "Come up with {} classification tasks. {} tasks are provided, and you need to propose {} different tasks. You don't need to solve these tasks.\n".format(machine_sample + seed_sample + machine_gen, machine_sample + seed_sample, machine_gen)
                for idx, ins in enumerate(prompt_instructions):
                    ins = re.sub(r"\s+", " ", ins).strip().rstrip(":")
                    prompt += f"Task {idx+1}. {ins}\n"
                prompt = {"role" : "user", "content" : prompt}
                prompts.append(prompt)
                if tt + 1 != round:
                    ans_ins = random.sample(machine_instructions, min(machine_sample, machine_gen))
                    ans_ins += random.sample(seed_instructions, machine_gen - len(ans_ins))
                    random.shuffle(ans_ins)
                    answer = ""
                    for idx, ins in enumerate(prompt_instructions):
                        ins = re.sub(r"\s+", " ", ins).strip().rstrip(":")
                        answer += f"Task {idx+machine_sample + seed_sample + 1}. {ins}\n"
                    answer = {"role" : "assistant", "content" : answer}
                    prompts.append(answer)
            # prompt += f"Task {len(prompt_instructions) + 1}. "
            
            print(f"prompt:\n{prompts}")
            try:
                print(prompts)
                completion = client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct:free",
                    messages=prompts, 
                    stop="\nTask {}.".format(machine_sample + seed_sample + machine_gen),
                    max_tokens = 1024,
                    n=1,
                    temperature=0.9,
                    top_p=0.7,
                )
                handler = handle_completion(completion, machine_gen)
                print(f"handler:{len(handler)}")
                for ins in handler:
                    all_instructions = seed_instructions + machine_instructions
                    scores = [scorer.score(ins, meta) for meta in all_instructions]
                    scores = [score["rougeL"].fmeasure for score in scores]
                    if max(scores) > 0.7:
                        print(f"{ins}, score:{max(scores)}")
                        continue
                    most_similar_instructions = {
                        all_instructions[i] : scores[i] for i in np.argsort(scores)[-10:][::-1]
                    }
                    machine_instructions.append(ins)
                    fw.write(json.dumps({
                        "instruction": ins,
                        "most_similar": most_similar_instructions,
                        "avg_similarity_score": float(np.mean(scores)),
                        "machine_req_idx": machine_req_idx
                    }) + "\n")
                    machine_req_idx += 1
                    fw.flush()
                print(machine_req_idx)
                time.sleep(5)
            except Exception as e:
                print(e)
                time.sleep(10)