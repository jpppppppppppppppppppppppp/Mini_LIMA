template=[
    {"role":"system","content":"Come up with an example for the following tasks. Write input and output seperately. If the task doesn't require input, write input None. Input is not needed for writing a code"},

]

template_cls=[
    # {"role":"system","content":"Given the classification task definition and the class labels. Write input and output seperately. If the task doesn't require input, write input None."},
    {"role":"system","content":"Given the classification task, write different possible Class label and corresponding Sample seperately"},

]

templete1=[
    {"role":"system","content":"Come up with examples for the following tasks. You can generate the output directly."},

    # {"role":"user","content":"Task: Which exercises are best for reducing belly fat at home?"},
    # {"role":"assistent",
    #  "content":"""Output:
    #             - Lying Leg Raises
    #             - Leg In And Out
    #             - Plank
    #             - Side Plank
    #             - Sit-ups"""},
   
    
    {"role":"user","content":"Task: Extract all the country names in the paragraph, list them separated by commas."},
    {"role":"assistent","content":"""
        Example 1
        Input: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favourably in the United States.
        Output: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.
    """},

    {"role":"user","content":"Task: Converting 85 F to Celsius."},
     {"role":"assistent","content":"Output: 85°F = 29.44°C"},

    # {"role":"user","content":"Task: Write a program to compute the sum of integers from k to n."},
    # {"role":"assistent","content":"""Output:
    #     def sum(k, n):
    #         sum = 0
    #         for i in range(k, n+1):
    #             sum += i
    #         return sum"""},
]

templete2=[
    {"role":"system","content":"Come up with examples for the following tasks. Try to generate multiple examples when possible."},

    {"role":"user","content":"Task: Extract all the country names in the paragraph, list them separated by commas."},
    {"role":"assistent","content":"""
        Example 1
        Paragraph: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favourably in the United States.
        Output: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.
    """},

    {"role":"user","content":"Task: Sort the given list ascendingly. "},
    {"role":"assistent","content":"""Example 1
        List: [10, 92, 2, 5, -4, 92, 5, 101]
        Output: [-4, 2, 5, 5, 10, 92, 92, 101]
        Example 2
        Input 2 - List: [9.99, 10, -5, -1000, 5e6, 999]
        Output: [-1000, -5, 9.99, 10, 999, 5e6]
     """},



]