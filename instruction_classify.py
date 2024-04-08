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

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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"
)

random.seed(time.time())

template = [
    {"role" : "system", "content" : "Can the following task be regarded as a classification task with finite output labels?"},
    
    {"role" : "user", "content" : "Task: Given my personality and the job, tell me if I would be suitable.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Give me an example of a time when you had to use your sense of humor.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Replace the placeholders in the given text with appropriate named entities.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Fact checking - tell me if the statement is true, false, or unknown, based on your knowledge and common sense.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Return the SSN number for the person.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Detect if the Reddit thread contains hate speech.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Analyze the sentences below to identify biases.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Select the longest sentence in terms of the number of words in the paragraph, output the sentence index.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Find out the toxic word or phrase in the sentence.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Rank these countries by their population.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: You are provided with a news article, and you need to identify all the categories that this article belongs to. Possible categories include: Music, Sports, Politics, Tech, Finance, Basketball, Soccer, Tennis, Entertainment, Digital Game, World News. Output its categories one by one, seperated by comma.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Given the name of an exercise, explain how to do it.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Select the oldest person from the list.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Find the four smallest perfect numbers.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Does the information in the document supports the claim? You can answer \"Support\" or \"Unsupport\".\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Create a detailed budget for the given hypothetical trip.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Given a sentence, detect if there is any potential stereotype in it. If so, you should explain the stereotype. Else, output no.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Explain the following idiom to me, and try to give me some examples.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Answer the following multiple choice question. Select A, B, C, or D for the final answer.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Decide whether the syllogism is logically sound.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: How can individuals and organizations reduce unconscious bias?\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: What are some things you can do to de-stress?\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Find out the largest one from a set of numbers. Output the number directly.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Replace the <mask> token in the text with proper words that are consistent with the context. You can use multiple words for each <mask> token.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Write a cover letter based on the given facts.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Identify the pos tag of the word in the given sentence.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: Write a program to compute the sum of integers from k to n.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: In this task, you need to compare the meaning of the two sentences and tell if they are the same. Output yes or no.\nIs it classification?"},
    {"role" : "assistant", "content" : "Yes"},

    {"role" : "user", "content" : "Task: To make the pairs have the same analogy, write the fourth word.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},

    {"role" : "user", "content" : "Task: Given a set of numbers, find all possible subsets that sum to a given number.\nIs it classification?"},
    {"role" : "assistant", "content" : "No"},
]

if __name__ == "__main__":
    idx = 1
    if os.path.exists(r"D:\Desktop\dsa\machine_tasks-{}.jsonl".format(idx)):
        with open(r"D:\Desktop\dsa\machine_tasks-{}.jsonl".format(idx), "r") as fw:
            lines = fw.readlines()
            print("Have loaded {} tasks".format(len(lines)))
    exist_ins = {}
    if os.path.exists(r"D:\Desktop\dsa\cls_machine_tasks-{}.jsonl".format(idx)):
        with open(r"D:\Desktop\dsa\cls_machine_tasks-{}.jsonl".format(idx), "r") as fw:
            for line in fw:
                ins = json.loads(line)
                exist_ins[ins["instruction"]] = ins
        print("Have existed {} tasks".format(len(exist_ins)))
    
    with open(r"D:\Desktop\dsa\cls_machine_tasks-{}.jsonl".format(idx), "w") as fw:
        i = 0
        while i + 1 < len(lines):
            line = json.loads(lines[i])
            if line["instruction"] in exist_ins:
                data = exist_ins[line["instruction"]]
                data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                fw.write(json.dumps(data, ensure_ascii=False) + "\n")
                i += 1
            else:
                prompt = "Task: " + line["instruction"].strip() + "\n" + "Is it classification?"
                prompt = {"role" : "user", "content" : prompt}
                prompts = template.copy()
                prompts.append(prompt)
                print(prompt["content"])
                try:
                    completion = client.chat.completions.create(
                        model="mistralai/mistral-7b-instruct:free",
                        messages=prompts, 
                        stop=["\n", "Task"],
                        max_tokens = 3,
                        n=1,
                        temperature=0,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    if completion.choices is not None:
                        print(completion.choices[0].message.content)
                        data = line
                        data["is_classification"] = completion.choices[0].message.content
                    else:
                        data["is_classification"] = ""
                    data = {
                            "instruction": data["instruction"],
                            "is_classification": data["is_classification"]
                        }
                    data = OrderedDict(
                            (k, data[k]) for k in \
                                ["instruction", "is_classification"]
                            )
                    fw.write(json.dumps(data, ensure_ascii=False) + "\n")
                    fw.flush()
                    i += 1
                    time.sleep(5)
                except Exception as e:
                    print(e)
                    time.sleep(10)
            if i % 5 == 4:
                print(i)