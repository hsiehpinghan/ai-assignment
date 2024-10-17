from quiz_generator import HistoryQuizGenerator, MathQuizGenerator
from schema import Quiz, Quizzes


history_test_case_1 = {
    "content": "Reformation",
    "keywords": ["Martin Luther", "Roman Catholic Church"],
}

history_test_case_2 = {"content": "World War II", "keywords": ["J. Robert Oppenheimer"]}

history_test_case_3 = {"content": "Civil War", "keywords": ["slavery"]}

math_test_case_1 = {}


def history_question(history_test_case: dict) -> Quiz:
    quiz_result = HistoryQuizGenerator().create_quiz(**history_test_case)
    print(f"\n\nQuiz: {quiz_result}\n\n")


def math_question(math_test_case: dict) -> Quizzes:
    quiz_result = MathQuizGenerator().create_quiz(**math_test_case)
    print(f"\n\nQuiz: {quiz_result}\n\n")


def bonus_point(history_test_case: dict, math_test_case: dict, num_quizzes: int):
    kwargs = {"num_quizzes": num_quizzes}
    quiz_result = HistoryQuizGenerator().create_quizzes(
        **{**history_test_case, **kwargs}
    )
    print(f"\n\nQuiz: {quiz_result}\n\n")
    quiz_result = MathQuizGenerator().create_quizzes(**{**math_test_case, **kwargs})
    print(f"\n\nQuiz: {quiz_result}\n\n")


if __name__ == "__main__":
    for history_test_case in [
        history_test_case_1,
        history_test_case_2,
        history_test_case_3,
    ]:
        history_question(history_test_case)

    math_question(math_test_case_1)

    # Bonus point is not necessary but a plus if you finish it.
    # bonus_point(history_test_case_1, math_test_case_1, 3)
