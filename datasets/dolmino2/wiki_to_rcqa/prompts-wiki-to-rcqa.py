DEFAULT ="""I am going to ask you to generate questions and answers about a passage. Here is the passage:

"{text}"

Generate {num_quest} that can be answered based on this passage.

Provide both the question and the answer. Use the format 

Question: ...
Answer: ...
"""

SPAN ="""I am going to ask you to generate questions and answers about a passage. Here is the passage:

"{text}"

Generate {num_quest} that can be answered based on this passage. Questions should not draw on facts outside of the passage. For each question, the answer should be a short phrase from the passage.

Provide both the question and the answer. Use the format 

Question: ...
Answer: ...
"""

PPHRASE ="""I am going to ask you to generate questions and answers about a passage. Here is the passage:

"{text}"

Generate {num_quest} that can be answered based on this passage. Questions should not draw on facts outside of the passage. For each question, the answer should be a short phrase from the passage. Questions should paraphrase and not use the exact words of the passage.

Provide both the question and the answer. Use the format 

Question: ...
Answer: ...
"""

DROP ="""I am going to ask you to generate questions and answers about a passage. Here is the passage:

"{text}"

Generate {num_quest} that can be answered based on this passage. Questions should not draw on facts outside of the passage. If appropriate try to use questions that require reasoning or calculations over the information in the passage. However, if the passage does not have information appropriate for reasoning or calculations, just ask other questions. 

The answer should be a span in the paragraph, a date, or a number.

Provide both the question and the answer. Don't include reasoning steps. Use the format 

Question: ...
Answer: ...
"""