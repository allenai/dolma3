### Overview 
Thid folder contains data documentations of the sources **verifiable-gpt41** and **verifiable-o4mini**.

Program-verifiable tasks are those that can use a (python) program to deterministically verify if an answer to a problem is correct or not. Solving these problems naturally requires a wide range of meta-reasoning strategies that are well-suited to be learned during a mid-training phase. We (1) programmatically generated these problems, (2) distilled reasoning traces from strong (reasoning) models (GPT-4.1 and o4-mini-high), and (3) finally filtered those for correctness using an output verifier (Python programs).

#### Question Generation

We manually curated 350 program-verifiable tasks, where each task consists of the following three components:
- Task Prompt Template. A natural language prompt format that defines the structure of task instances, with placeholders for instance-specific parameters. Examples of "task prompt templates" can be found in `example_task_templates` folder.
- Parameter Generator. A Python program that samples valid instance parameters based on user-defined constraints, such as scale or structure.
- Output Verifier. A Python program that evaluates the model’s output, checking whether or not the output is correct.

#### Reasoning and Answer Generation

For each generated, we generated reasoning traces along with the final answer using GPT-4.1 and o4-mini with high reasoning level. Reasoning and answers were generated using two prompting methods: *simple prompting* (provided in `response_generation_prompts/simple_prompting.txt`), and *priming with meta reasoning strategies* (provided in `response_generation_prompts/priming_w_meta_reasoning_prompting.txt`).

Finally, we entirely removed `<think> </think>` tags, and replaced `<answer> </answer>` with just `Answer: …`.

#### Filtering by Verification
As described above, each task has an output verifier, a Python program that evaluates the model’s output, checking whether or not the output is correct. In the SAT example (`example_task_templates/SAT.txt`), the output verifier checks two things: (1) whether the format of the output is valid (i.e., exactly N binary values), and (2) how many clauses are satisfied by the model output's assignment; if the format is valid, the verifier checks whether all clauses are satisfied - an output will pass the verification if yes. We run each task’s verifier on every instance’s outputs, and keep those that are verified to be correct.