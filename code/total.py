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
from collections import OrderedDict
def sum_up_ins():
    total = {}
    for idx in range (1,2):
        if os.path.exists(r"D:\Desktop\dsa\machine_tasks-{}.jsonl".format(idx)):
                with open(r"D:\Desktop\dsa\machine_tasks-{}.jsonl".format(idx), "r", encoding="utf-8") as fw:
                    for line in fw:
                        data = json.loads(line)
                        ins = data['instruction']
                        total[ins] = data

    for idx in range (1,2):
        if os.path.exists(r"D:\Desktop\dsa\cly_machine_tasks-{}.jsonl".format(idx)):
                with open(r"D:\Desktop\dsa\cly_machine_tasks-{}.jsonl".format(idx), "r", encoding="utf-8") as fw:
                    for line in fw:
                        data = json.loads(line)
                        ins = data['instruction']
                        total[ins] = data
    seed_tasks = [json.loads(line) for line in open(r"D:\Desktop\dsa\seed_tasks.jsonl", "r", encoding="utf-8")]
    written = [json.loads(l)["instruction"] for l in open(r"D:\Desktop\dsa\machine-task.jsonl", encoding="utf-8")]
    to_write = [json.loads(l) for l in open(r"D:\Desktop\dsa\machine-task.jsonl", encoding="utf-8")]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    with open(r"D:\Desktop\dsa\machine-task.jsonl", "w", encoding="utf-8") as fw:
        for ins in total:
            data = total[ins]
            if ins in written:
                fw.write(json.dumps(data) + '\n')
                continue
            all_instructions = seed_tasks + to_write
            scores = [scorer.score(ins, meta['instruction']) for meta in all_instructions]
            scores = [score["rougeL"].fmeasure for score in scores]
            if max(scores) > 0.7:
                print(f"{ins}, score:{max(scores)}")
                continue
            to_write.append(data)
            fw.write(json.dumps(data) + '\n')
            if len(to_write) % 100 == 0:
                print(len(to_write))
        fw.flush()

def sum_up_cly():
    ins = [json.loads(line) for line in open(r"D:\Desktop\dsa\machine-task.jsonl", "r", encoding="utf-8")]

    cly = {}
    for i in range (1,2):
        if os.path.exists(r"D:\Desktop\dsa\cls_machine_tasks-{}.jsonl".format(i)):
            with open(r"D:\Desktop\dsa\cls_machine_tasks-{}.jsonl".format(i), "r", encoding="utf-8") as fw:
                for line in fw:
                    data = json.loads(line)
                    cly[data['instruction']] = data

    for i in range (1, 2):
        if os.path.exists(r"D:\Desktop\dsa\cly_cls_machine_tasks-{}.jsonl".format(i)):
            with open(r"D:\Desktop\dsa\cly_cls_machine_tasks-{}.jsonl".format(i), "r", encoding="utf-8") as fw:
                for line in fw:
                    data = json.loads(line)
                    cly[data['instruction']] = data

    with open(r"D:\Desktop\dsa\machine-task-cly.jsonl","w", encoding="utf-8") as fw:
        for i in ins:
            if i["instruction"] in cly:
                i["is_classification"] = cly[i["instruction"]]["is_classification"]
            else:
                i["is_classification"] = "None"
            fw.write(json.dumps(i) + '\n')
            fw.flush()

def sum_up_exa():
    ins = [json.loads(line) for line in open(r"D:\Desktop\dsa\machine-task-cly.jsonl", "r", encoding="utf-8")]
    jsonl_files = []
    for root, dirs, files in os.walk(r"D:\Desktop\dsa\instance"):
        for filename in files:
            if filename.endswith("jsonl"):
                jsonl_files.append(os.path.join(root, filename))
    instance = {}
    for file in jsonl_files:
        with open(file, "r", encoding="utf-8") as fw:
            for line in fw:
                data = json.loads(line)
                if data["instruction"] in instance:
                    instance[data['instruction']]["instance"].append({"input":data["input"], "output":data["output"]})
                else:
                    data["instance"] = [{"input":data["input"], "output":data["output"]}]
                    instance[data['instruction']] = data
    exist = {json.loads(line)["instruction"]:json.loads(line) for line in open(r"D:\Desktop\dsa\machine-task-ins.jsonl", "r")}
    with open(r"D:\Desktop\dsa\machine-task-ins.jsonl", "w", encoding="utf-8") as fw:
        for i in ins:
            if i["instruction"] in exist:
                fw.write(json.dumps(exist[i["instruction"]])+'\n')
                continue
            handler = OrderedDict((k, i[k]) for k in ["instruction", "is_classification"])
            handler["cly_error"] = "False"
            if i["instruction"] in instance:
                if i["is_classification"] == instance[i["instruction"]]["is_classification"]:
                    handler["instance"] = instance[i["instruction"]]["instance"]
                else:
                    handler["instance"] = instance[i["instruction"]]["instance"]
                    i["is_classification"] = instance[i["instruction"]]["is_classification"]
                    handler["cly_error"] = "True"
            else:
                handler["instance"] = []
                continue
            fw.write(json.dumps(handler)+'\n')
        fw.flush()


def check_sum():
    sum_up_ins()
    sum_up_cly()
    sum_up_exa()
    exist = [json.loads(line) for line in open(r"D:\Desktop\dsa\machine-task-ins.jsonl", "r", encoding="utf-8")]
    cly = 0
    ins = 0
    cly_emp = 0
    ins_emp = 0
    cly_err = 0
    with open(r"error.jsonl", "w") as fw:
        for i in exist:
            if i["is_classification"] != "None":
                cly += 1
            else:
                cly_emp += 1
            ins += len(i["instance"])
            if len(i["instance"]) == 0:
                ins_emp += 1
            if i["cly_error"] == "True":
                fw.write(json.dumps(i) + '\n')
                cly_err += 1
        fw.flush()
    print(len(exist), cly, ins)
    print(cly_emp, ins_emp, cly_err)

check_sum()
