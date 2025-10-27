############ Verbosity-reducing Rewrites ############

FULL_THOUGHTS_PROMPT = """I will provide a math problem and a long solution that arrives at the correct answer. Please write a cleaned version preserving ALL reasoning, explanations, and thoughts present in the draft.

Instructions for the write-up:
- Do NOT omit or skip any exploratory steps, internal questions, or musings (such as “maybe I can try this...”, “hmm, is that right?”, “wait, let’s check a case”, “this seems off”, etc.).
- Include every approach I tried, even if I abandoned it or found it less useful.
- Write as a continuous narrative with all the “thinking out loud,” doubts, and thought process included, not just a polished proof.
- Do not include any separate formatting, beyond the cleaned solution.
- Use proper LaTeX syntax for mathematical expressions.
- Only edit for clarity, flow, and formatting (such as adding LaTeX); do not summarize, condense, or remove any part of my reasoning.
- End the solution with: "The final answer is <answer>."
"""


EMPHASIZE_META_REASONING_PROMPT = """You are given:
- A math problem and its original solution trace which contains meta-reasoning behaviors.
- A list of three meta-reasoning behaviors (from the following list):

1. Problem Reformulation
2. Strategic Method Planning or Selection
3. Backward Chaining (Goal-Driven Reasoning)
4. Testing, Counterexamples, and Sanity Checks
5. Self-Verification and Cross-Checking with Different Methods
6. Noticing, Correcting, or Abandoning Errors/Dead Ends (Backtracking)
7. Subgoal Identification and Decomposition

**Your task:**

Rewrite the solution trace so that it exemplifies the following three meta-reasoning behaviors throughout the reasoning:

- {behavior1}
- {behavior2}
- {behavior3}

The rewritten solution should follow the exact same flow of the original solution. You may add natural, plausible self-talk to highlight the behaviors, but do not significantly change the style or formatting of the original solution. You may otherwise condense irrelevant reasoning that is not directly related to the three behaviors as appropriate.

**Requirements:**
- The rewritten solution must be detailed, well-structured, and contain high-quality, correct LaTeX math.
- Emphasize the three behaviors, but keep the reasoning flow the same as the original solution.
- Do not just list the behaviors, but rather demonstrate them through the internal thoughts.
- Do not include section headers listing the behaviors.
- IMPORTANT: Do not add any new steps or reasoning that is not already present in the original solution trace. The goal is to rewrite the existing reasoning to showcase meta-reasoning behaviors, not to add any new content.
- End your answer with the final answer in LaTeX format: \\boxed{{Your answer}}.
---
**Problem:**

{problem}

**Original Solution:**

{solution_trace}
---
"""


SLEEK_PROMPT = """I will provide a math problem and a long or rough solution that arrives at the correct answer. Please write a clean, logically complete, and step-by-step final solution that contains all and only the reasoning steps that are **relevant** to the correct answer (omit tangents, dead ends, and unnecessary explorations).

Instructions for the write-up:
- Do not include any separate formatting, beyond the cleaned solution.
- Do not include any additional reasoning that is not present in the original draft.
- Include only the steps and justifications that are required to arrive at the answer, even if the draft solution contains extra exploration or false starts.
- Motivate and explain any method or approach that is necessary to reach the solution.
- Use proper LaTeX syntax for mathematical expressions.
- Write the solution as if teaching a student—clear, concise, and logically complete.
- End the solution with: "The final answer is <answer>."
"""
####################################################################################

############ Dialogue Rewrites ############



STUDENT_TEACHER_LECTURE = """Convert the context above as a multi-turn discussion between a teacher and an astute student. To do so, first list the key techniques and concepts used in the solution. Then, synthesize a conversation where the teacher first gives the student a extensive and detailed lecture on all concepts and related topics. The student asks intelligent follow up questions to clarify their understanding. The teacher then poses the given problem to test the student, and guides the student to the solution by asking socratic questions to help the student solve the problem step-by-step. Make sure that the student's reasoning when solving the problem strictly adheres to the solution above and remains faithful to information in the original solution. Please DO NOT add any new reasoning steps in the student's solution other than what is present in the context.

Additional instructions:
-- The last dialogue turn should be the student giving the final answer in \\boxed format.
-- All math should be in LaTeX format.
"""

STUDENT_TEACHER_PLANNING = """Convert the context above as a multi-turn discussion between a teacher and a student. The teacher first poses the question, lists possible techniques to reach the solution, and then guides the student to the solution by asking socratic questions to help the student solve the problem step-by-step. Make sure that the student's reasoning when solving the problem strictly adheres to the solution above and remains faithful to information in the original solution. Please DO NOT add any new reasoning steps in the student's solution other than what is present in the context.

Additional instructions:
-- The last dialogue turn should be the student giving the final answer in \\boxed format.
-- All math should be in LaTeX format."""


STUDENT_TEACHER_REFORMULATION = """Convert the context above as a multi-turn discussion between a teacher and a student. The teacher poses the question in the first dialouge turn. The student then reformulates the problem, and afterwards the teacher guides the student to the solution by asking socratic questions to help the student solve the problem step-by-step. Make sure that the student's reasoning when solving the problem strictly adheres to the solution above and remains faithful to information in the original solution. Please DO NOT add any new reasoning steps in the student's solution other than what is present in the context.

Additional instructions:
-- The last dialogue turn should be the student giving the final answer in \\boxed format.
-- All math should be in LaTeX format."""


STUDENT_STUDENT_ERROR_CORRECT = """Convert the context above as a multi-turn discussion between two students who are working on the given problem. The students ask each other questions to solve the problem step-by-step. When one student makes an error, the other student identifies the issue and prompts the original student to correct it. Make sure that their discussion strictly adheres to the context above and remains faithful to information in the context. The students' errors should be based on the errors made and corrected in the original solution trace. Please DO NOT add any new information/reference other than the context.

Additional instructions:
-- The last dialogue turn should be the student giving the final answer in \\boxed format.
-- All math should be in LaTeX format."""

