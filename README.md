## Junior Generative AI Engineer Assignment
 

### Task 1: History Quiz
- Generate a multiple-choice question. The subject is history.
- The quiz must be related to the given `content` and the provided `keywords`.
- Actions
    1. Implement the HistoryQuizGenerator in quiz_generator.py
    2. Use GPT-3.5 with [LangChain](https://python.langchain.com/docs/get_started/introduction) to implement the create_quiz function.
    3. Run main.py to evaluate three history_test_cases.

### Task 2: Math Quiz
- Generate a math word quiz (應用題) about a two-variable linear system (二元一次方程式).
- Actions
    1. Implement the MathQuizGenerator in quiz_generator.py.
    2. Use GPT-3.5 with [LangChain](https://python.langchain.com/docs/get_started/introduction) to implement the create_quiz function.
    3. Run main.py to evaluate one math_test_case.

### Bonus Points
1. Write an `error handling mechanism` to ensure that, regardless of the input, the output is always in a valid format.
2. Implement the create_quizzes functions in the HistoryQuizGenerator and MathQuizGenerator to ensure that quizzes can be generated `in parallel`.

### Note
- Check /sample_quiz folder for the example output.
- Please read each file properly. All the information you need is included in this directory. If you believe some information is required but not provided, you can decide yourself but write comments above the code.
- You need to use Python 3.11.