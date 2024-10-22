from error import ErrorHandler
from schema import Quiz, Quizzes
from quiz_generator import MathQuizGenerator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain.output_parsers import PydanticOutputParser

OUTPUT_EXAMPLE = '''
{
  "question": "A group of 41 people are going to a concert together in 7 vehicles. Some of the vehicles can hold 7 people each, and the rest can hold 5 people each. Assuming all the vehicles are filled to capacity, exactly how many of the vehicles can hold 7 people?",
  "options": [
    {
      "content": "If 1 vehicle can hold 7 people, then 7 - 1 = 6 vehicles can hold 5 people.",
      "reason": "5*6 + 7*1 != 41",
      "isCorrect": false
    },
    {
      "content": "If 2 vehicles can hold 7 people, then 7 - 2 = 5 vehicles can hold 5 people.",
      "reason": "5*5 + 7*2 != 41",
      "isCorrect": false
    },
    {
      "content": "If 3 vehicles can hold 7 people, then 7 - 3 = 4 vehicles can hold 5 people.",
      "reason": "5*4 + 7*3 = 41",
      "isCorrect": true
    },
    {
      "content": "If 4 vehicles can hold 7 people, then 7 - 4 = 3 vehicles can hold 5 people.",
      "reason": "5*3 + 7*4 != 41",
      "isCorrect": false
    }
  ]
}
'''.strip()

USER_PROMPT = '''
Generate a math word quiz involving a two-variable linear system in the form of a question with multiple choice options.

The quiz should include the following elements:
- A word problem that involves a two-variable linear system.
- Four answer options, each with a described reasoning and a correctness indicator.

# Steps

1. **Problem Creation**: Develop a real-world scenario that can be modeled using a two-variable linear system. Ensure clarity and the logical flow of the question.
2. **Solution Identification**: Calculate the correct solution to the problem, ensuring there is one unique solution.
3. **Option Development**: Create four answer options. Three incorrect, each accompanied by a reason showing they are incorrect, and one correct with a supporting explanation.
4. **Verification**: Ensure the problem and the given options have logical coherence and consistency, further explaining why the incorrect options don't satisfy the conditions.

# Output Format

A JSON object with the following structure:
- `question`: A string that states the math word problem.
- `options`: An array of objects, each representing a possible answer.
  - Each option object should contain:
    - `content`: A string describing the potential solution.
    - `reason`: A string explaining why the option is correct or incorrect.
    - `isCorrect`: A boolean indicating if the option is correct.

# Example Output

```json
{output_example}
```

# Notes

- Ensure that the word problem is understandable and applicable to a real-world context.
- Each option must accurately reflect the underlying mathematics of solving a two-variable linear equation.
'''.strip()

class MyMathQuizGenerator(MathQuizGenerator):
    def __init__(self):
        self.output_example = OUTPUT_EXAMPLE
        self.user_prompt = USER_PROMPT
        self.parser = PydanticOutputParser(pydantic_object=Quiz)
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0)
        self._error_handling = ErrorHandler()

    def create_quiz(self) -> Quiz:
        chain = self._create_chain()
        quiz = chain.invoke(self._get_chain_input())
        return quiz

    def create_quizzes(self, num_quizzes: int) -> Quizzes:
        sub_chain = self._create_chain()
        chain = RunnableParallel(**{f'quiz_{i}': sub_chain for i in range(num_quizzes)})
        result = chain.invoke(self._get_chain_input())
        quizzes = Quizzes(quizzes=[result[k] for k in result])
        return quizzes

    def _get_chain_input(self):
        chain_input = {'output_example': self.output_example}
        return chain_input

    def _create_chain(self):
        chain = PromptTemplate.from_template(template=self.user_prompt) | self.model | self.parser
        chain_with_error_handling = chain.with_fallbacks(fallbacks=[self._error_handling | self.parser],
                                                         exception_key="exception")
        return chain_with_error_handling

