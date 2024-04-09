import os
import json
import jsonlines
import random
import tqdm
import re
import string
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from openai import OpenAI
from collections import OrderedDict

from instance_gen_template import output_first_template_for_clf, input_first_template_for_gen
from prompts import templete1,template

import time

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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"
#   api_key="fLXHlcFnFjoNctz5Y3QS296AxEcae121"
)

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
        raw = re.split(r"\n?Task \d+\s?\. ", cho)
        ret = []
        for ins in raw:
            ins = re.sub(r"\s+", " ", ins).strip()
            ins = ins.strip().capitalize()
            if ins == "":
                continue
            if len(ins.split()) <= 3 or len(ins.split()) > 150:
                continue
            if any(find_word_in_string(word, ins) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to", "music"]):
                continue
            # if ins.startswith("Write a program"):
            #     continue
            if not ins[0].isascii():
                continue
            if ins[0].isnumeric():
                continue
            if ins[0] in string.punctuation:
                continue
            ret.append(ins)
        return ret
    return []
    
batch_dir="F:\workspace\hw2"
# input_file="machine_tasks-2.jsonl"
input_file="dsadsa\cls_machine_tasks-1.jsonl"
output_file="data\instance-1.jsonl"

classification_tasks_only=False
generation_tasks_only=False
max_instances_to_generate=5
request_batch_size=1


if __name__ == '__main__':
   

    with open(os.path.join(batch_dir, input_file)) as fin:
        lines = fin.readlines()
    
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)
            # break
        
    output_path = os.path.join(batch_dir, output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    # with open(output_path, "w") as fout:
    with jsonlines.open(output_path,"a") as fout:
            
        for batch_idx in range(0, len(tasks), request_batch_size):
            batch = tasks[batch_idx: batch_idx + request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                continue
            
            else:
                for task in batch:
                    if task["is_classification"]=="Yes":
                        continue
                 
                    prompt =   "Task: " + task["instruction"].strip() + "\n"
                    print(prompt)
                    prompts = {"role" : "user", "content" : prompt}
                    try:
                        results = client.chat.completions.create(
                            model="mistralai/mistral-7b-instruct:free",
                            # model="mistralai/Mistral-7B-Instruct-v0.2",
                            messages=template+[prompts], 
                            # stop="\nTask {}.".format(machine_sample + seed_sample + machine_gen),
                            stop=[f"Example {max_instances_to_generate + 1}", "Task:"],
                            max_tokens = 512,
                            n=1,
                            temperature=1, #文本更加多样化
                            top_p=0.5,
                        )
                    
                        model_result=results.choices[0].message.content
                        print(model_result)
                        data = task
                        pattern = re.compile(r'Input:(.*?)Output:(.*?)(Input:|$)', re.DOTALL)
                        matches = pattern.search(model_result)
                        data["input"]=matches.group(1).strip()
                        data["output"]=matches.group(2).strip()
                        
                        fout.write(data)
                        progress_bar.update(len(batch))
                        time.sleep(5)

                    except Exception:
                        time.sleep(10)

               