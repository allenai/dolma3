### Overview

This folder contains details about the data creation process of source **math-meta-reasoning**.

#### Capability Selection and Task Design
We begin by identifying structured reasoning capabilities that are critical for mathematical problem-solving. Inspired by human structured reasoning, we select seven core capabilities that research indicates are foundational to mathematical expertise: self-awareness, strategy selection, self-evaluation, goal management, hierarchical organization, backward chaining, and conceptual reasoning. We then design specific tasks that systematically target these capabilities, as shown in the Table below. 


| Task | Capabilities |
|------|--------------|
| 1. Math error recovery | Self-awareness, verification, backtracking |
| 2. Choosing the **technique** to use | Strategy selection |
| 3. Difficulty estimation + add self-awareness prompts | Self-evaluation |
| 4. Steps generation | Goal management, hierarchical organization |
| 5. From answer, generate steps backwards | Backward chaining |
| 6. Conversation generation | All capabilities (tagging) |
| 7. Reason about necessary concepts and how they connect | Conceptual reasoning |


#### Dataset Selection and Automatic Annotation

[Pandalla-math](https://huggingface.co/datasets/pandalla/pandalla-math-dataset-v1.0) was used as our seed dataset due to its comprehensive metadata annotations that provide natural scaffolding for capability-targeted data generation. Each problem includes detailed annotations covering problem classification, difficulty analysis, solution approaches, common pitfalls, and verification methods. These rich annotations serve as natural inputs for our capability-targeted tasks. For example, the "common_pitfalls" field directly informs Math Error Recovery generation, while "solution_approach.steps" provides structure for backward chaining tasks. Our specific usage of these annotations can be found in the task prompts.

To scale beyond Pandalla-math's 2,000 problems, we use o4-mini to automatically generate similar metadata annotations for [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) (40.3K problems) and [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) (127.7K problems) \citep{moshkov2025aimo2}.


#### Large-Scale Data Generation
Using the annotated datasets as foundation, we employ GPT-4.1 and o4-mini to generate training data at scale for each capability-targeted task. Prompts for each task can be found in `task_prompts` folder. 


