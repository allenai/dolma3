### Overview

This folder contains details about the data creation process of source **code-meta-reasoning**.

#### Capability Selection and Task Design
For code reasoning, we identify structured reasoning capabilities critical for software problem-solving. Inspired by human structured reasoning, we select seven core capabilities that research indicates are foundational to programming expertise: self-awareness, strategy selection, self-evaluation, goal management, hierarchical organization, backtracking/debugging, and conceptual reasoning. 

We then design specific code tasks that target these capabilities, shown in the Table below. 



| Task                                | Capabilities                                         |
|-------------------------------------|------------------------------------------------------|
| Code error recovery (single-turn)   | Self-awareness, verification, backtracking           |
| Code error recovery (multi-turn)    | Self-awareness, verification, backtracking           |
| Planning the solution               | Strategy selection, goal management                  |
| Solution implementation             | Conceptual-level processing, hierarchical organization |
| Code quality evaluation (high/low)  | Self-evaluation                                      |
| Difficulty estimation               | Self-evaluation, self-awareness                      |
| Unit test walkthrough               | Goal management, verification                        |



#### Dataset Selection and Generation

For code generation capabilities, we use [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) and [tulu-3-sft-personas-code](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code) to generate data. Inspired by the annotation categories in the [Pandalla-math](https://huggingface.co/datasets/pandalla/pandalla-math-dataset-v1.0) dataset (which we use for our math-meta-reasoning source), we first use o4-mini to generate comprehensive metadata annotations, in categories of problem classification, difficulty analysis, solution approaches, common implementation pitfalls, and code verification methods. These annotations are then inserted as supplementary information in the task prompts -- see the task prompts for specific annotation types used by task. Using these prompts, we generate data for each task using GPT-4.1. 

Prompts for each task can be found in `task_prompts` folder.


