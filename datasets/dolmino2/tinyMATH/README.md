# TinyMATH Data

This outlines the steps taken to create the TinyMATH PoT and TinyMATH MIND datasets. Broadly speaking, TinyMATH is a collection of synthetic high-school level mathematics problems, in the spirit of tinyGSM.

## Overview
For the TinyMATH datasets we generate new math problems, code to solve them, and two forms of natural language discussion of each solution.  Generation of this data proceeds in three steps:
1. **Problem Generation**: Generate 100 synthetic math problems from each problem in the MATH training dataset
2. **Code Generation**: Convert new problems to python solutions (PoT)
3. **Conversational Transformation**: Convert python solutions to natural language discussions
While generating new problems is a critical step in this pipeline, we only include the generated solutions, both in code and conversational formats, in our training data.

## Source Materials

- **Base Dataset**: [Hendrycks MATH Dataset](https://github.com/hendrycks/math) (7,500 training examples)
- **Inspiration**: [tinyGSM protocol](https://arxiv.org/abs/2312.09241), [MIND rewrites](https://arxiv.org/pdf/2410.12881)

## Stage 1: Problem Generation

The first step is to generate 100 math problems from each individual example in the MATH training dataset. This is done using the Azure API and OpenAI models as follows:

**Model**: `gpt-4.1-2025-04-14`

**Prompt**:
```
Design 100 high-school level math problems similar to a given math problem.
- Make sure the language is different and sufficiently rephrased.
- Make sure the numbers are different from the original problem.
- Feel free to imbue this as a word problem and manufacture stories, characters, and scenarios.
- If you create a word problem, make sure that the problem is correct and does not say anything incorrect when phrasing the problem.
- Make sure the new problems are no easier than the given problem and require more steps to solve.
- Make sure the new problems are self-complete, clear, unambiguous, and coherent.
- Please do not include hints or solutions for the new problems.
- Start each new question with Problem X (where X is a number) and end each one with <|endofproblem|>
- Stop generating new questions when you generate 100 questions.
- Make sure you generate EXACTLY 100 questions.

## The original math problem
%s

## New problems
- Problem 1:
```

**Output**: 
- 729,888 problems (7,500 × 100, minus some failures in generation/parsing)


## Stage 2: Program of Thought (PoT) Generation

The next step is to follow the spirit of tinyGSM and write pythonic solutions for each of the new problems. This is done using the Azure API and OpenAI models as follows:

**Model**: `gpt-4o-mini-2024-07-18`

**Prompt**:
```
For a given math problem write down a **detailed and complete Python program** to solve the question **step by step**.
- Do NOT give the result directly
- Do NOT write any calculations in the comments. Some reasoning in the comments is alright, but do NOT put any numerical calculations in the comments.
- The program should contain multiple lines of code and end with 'result = XXX' and then 'return result' (Make sure to replace XXX with the actual result of the python program).
- Make sure your Python program is complete and solves the problem. Do **NOT** write things like 'solution to be completed', result = ?, insert your code here etc.
- Give the complete solution to solve the problem, written in Python. Do not write things like 'insert your code here'.
- Each program should end with <|endofprogram|>.
- Include the problem as given to you as the docstring for the python function.
- The Python program should be named "simple_math_problem" and take no arguments
- Do not import anything

The original problem:
%s
```

**Output**: 729,261 python solutions (1 per problem, minus some failures in generation)

## Stage 3: Conversational Transformation
The [MIND paper](https://arxiv.org/pdf/2410.12881) proposes 8 templates for rewriting math problems into conversational english. However they demonstrate that two rewrite-templates: the "Two Students" and "Problem Solving" templates, demonstrate superior performance. This finding was corroborated in [OLMo2](https://arxiv.org/abs/2501.00656) in the TinyGSM dataset. Hence, for TinyMATH, we only rewrite the PoT data using these two templates.

**Model**: `gpt-4.1-2025-04-14`

**Two Students Prompt**:
```
%s
Convert the context above as a multi-turn discussions between two students who are working on their assignment related to the given context. Make sure that their discussions strictly adhere to the context above and remains faithful to information in the context. If there are any mathematical calculations that need to be performed, please perform them. Other than that, please DONOT add any new information/reference other than the context. DONOT assume the ability to call any code or tools.
```

**Problem Solving Prompt**:
```
%s 
Convert the context above as a multi-turn problem-solving conversation where participants analyze challenges or scenarios presented in the content and brainstorm solutions within the context of the provided material, avoiding speculation or unrelated discussions. Make sure that their conversation strictly adhere to the context above and remains faithful to information in the context. If there are any mathematical calculations that need to be performed, please perform them. Other than that, please DONOT add any new information/reference other than the context. DONOT assume the ability to call any code or tools.
```

**Output**: 1,413,080 natural language solutions (2 per python solution, minus some failures in generation)

## Final Dataset
- **TinyMATH PoT**: 729,261 python solutions | 241M tokens
- **TinyMATH MIND:**: 1,413,080 natural language solutions | 899M tokens
