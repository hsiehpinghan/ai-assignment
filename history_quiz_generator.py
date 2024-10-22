import json
from typing import List

from error import ErrorHandler
from schema import Quiz, Quizzes
from quiz_generator import HistoryQuizGenerator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain.output_parsers import PydanticOutputParser

USER_PROMPT = '''
{input}
'''.strip()

INPUT_EXAMPLE = '''
{
  "content": "Lecompton Constitution",
  "keywords": ["Missouri"]
}
'''.strip()

OUTPUT_EXAMPLE = '''
{
  "question": "Who created the proslavery Lecompton Constitution?",
  "options": [
    {
      "content": "Mexico; after its adoption Texas broke away and joined the United States so it could be a free state",
      "reason": "Mexico had no involvement in the Lecompton Constitution!",
      "isCorrect": false
    },
    {
      "content": "\"Border ruffians\" from Missouri who crossed the border to vote for the legalization of slavery in Kansas",
      "reason": "These new residents helped secure a proslavery legislature which drafted a proslavery constitution known as the Lecompton Constitution.",
      "isCorrect": true
    },
    {
      "content": "The Confederate States of America, who adopted the Lecompton Constitution in lieu of the Constitution of the United States",
      "reason": "The Lecompton Constitution was not connected with the Confederate States of America, which did not yet exist.",
      "isCorrect": false
    },
    {
      "content": "Kansas settlers who voted to oppose the imposition of slavery in their state",
      "reason": "While many Kansas settlers opposed slavery, the Lecompton Constitution was created by proslavery forces, not those who opposed it.",
      "isCorrect": false
    }
  ]
}
'''.strip()

SYSTEM_PROMPT = '''
Generate a multiple-choice question where multiple correct choices are possible, focusing on the subject of history. The quiz should relate to the provided `content` and `keywords`.

Ensure the quiz comprises a question and four answer options, grounded in the provided data.

# Steps

1. Analyze the `content` and `keywords` to understand the historical context and relevant information.
2. Formulate a question that allows for multiple correct answers, related to the history context provided.
3. Create four potential answer options:
   - Use historical facts to create plausible wrong options and correct options.
   - Ensure each option includes a reason explaining its correctness or incorrectness.
   - Identify if an option is correct or incorrect.

# Output Format

Produce a JSON including:
- `question`: A string representing the content of the question.
- `options`: An array of four objects, each containing:
  - `content`: The text of the option.
  - `reason`: A clear explanation of why the option is correct or incorrect.
  - `isCorrect`: Boolean indicating whether the option is correct.

# Examples

## Input
```json
{input_example}
```

## Output
```json
{output_example}
```

# Notes

- Ensure each question and option are historically accurate and derived from the "content" and "keywords."
- The question must be structured to have at least one correct option.
'''.strip()

class MyHistoryQuizGenerator(HistoryQuizGenerator):
    def __init__(self):
        self.user_prompt = USER_PROMPT
        self.input_example = INPUT_EXAMPLE
        self.output_example = OUTPUT_EXAMPLE
        self.system_prompt = SYSTEM_PROMPT
        self.parser = PydanticOutputParser(pydantic_object=Quiz)
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0)
        self._error_handling = ErrorHandler()

    def create_quiz(self, content: str, keywords: List[str]) -> Quiz:
        chain = self._create_chain()
        quiz = chain.invoke(self._get_chain_input(input={'content': content, 'keywords': keywords}))
        return quiz

    def create_quizzes(
        self, content: str, topics: List[str], num_quizzes: int
    ) -> Quizzes:
        sub_chain = self._create_chain()
        chain = RunnableParallel(**{f'quiz_{i}': sub_chain for i in range(num_quizzes)})
        result = chain.invoke(self._get_chain_input(input={'content': content, 'keywords': topics}))
        quizzes = Quizzes(quizzes=[result[k] for k in result])
        return quizzes

    def _get_chain_input(self, input: dict):
        input = json.dumps(input,
                           ensure_ascii=False,
                           indent=2)
        chain_input = {'input': input,
                       'input_example': self.input_example,
                       'output_example': self.output_example}
        return chain_input

    def _create_chain(self):
        chain = ChatPromptTemplate.from_messages([("system", self.system_prompt),
                                                  ("user", self.user_prompt)]) | self.model | self.parser
        chain_with_error_handling = chain.with_fallbacks(fallbacks=[self._error_handling | self.parser],
                                                         exception_key="exception")
        return chain_with_error_handling

