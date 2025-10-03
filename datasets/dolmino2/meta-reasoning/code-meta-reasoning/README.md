### Overview

This folder contains details about the data creation process of source **code-meta-reasoning**.

#### Capability Selection and Task Design
Following the same approach, for code reasoning, we identify structured reasoning capabilities critical for software problem-solving. From the taxonomy of human structured reasoning, we select seven core capabilities that research indicates are foundational to programming expertise: self-awareness, strategy selection, self-evaluation, goal management, hierarchical organization, backtracking/debugging, and conceptual reasoning. 

We then design specific code tasks that target these capabilities, shown in Table below. 



| Task                                | Capabilities                                         |
|-------------------------------------|------------------------------------------------------|
| Code error recovery (single-turn)   | Self-awareness, verification, backtracking           |
| Code error recovery (multi-turn)    | Self-awareness, verification, backtracking           |
| Planning the solution               | Strategy selection, goal management                  |
| Solution implementation             | Conceptual-level processing, hierarchical organization |
| Code quality evaluation (high/low)  | Self-evaluation                                      |
| Difficulty estimation               | Self-evaluation, self-awareness                      |
| Unit test walkthrough               | Goal management, verification                        |



#### Dataset Selction and Generation

For code generation capabilities, we use [APPS](https://huggingface.co/datasets/codeparrot/apps), [TACO](https://huggingface.co/datasets/BAAI/TACO), [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning), and [tulu-3-sft-personas-code](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code) to generate data. In particular, APPS and TACO are subsets of OpenCodeReasoning. Similar to our approach with math-meta-reasoning dataset, we use o4-mini to generate comprehensive metadata annotations for these coding datasets. Each problem includes annotations covering problem classification, difficulty analysis, solution approaches, common implementation pitfalls, and code verification methods. We then use these annotations to generate data for each task using GPT-4.1. 

Prompts for each task can be found in `task_prompts` folder.
