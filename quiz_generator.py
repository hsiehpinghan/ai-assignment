from abc import ABC, abstractmethod
from typing import List
from schema import Quiz, Quizzes


class HistoryQuizGenerator(ABC):
    """
    A quiz generator generating history quiz based on the given content and keywords.
    """

    @abstractmethod
    def create_quiz(self, content: str, keywords: List[str]) -> Quiz:
        """
        Create a quiz including its question, and four options. Based on the given content and keywords.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass

    @abstractmethod
    def create_quizzes(
        self, content: str, topics: List[str], num_quizzes: int
    ) -> Quizzes:
        """
        Create multiple quizzes.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass


class MathQuizGenerator(ABC):
    """
    A quiz generator generating math word quiz which involves a linear equation with two variables.
    """

    @abstractmethod
    def create_quiz(self) -> Quiz:
        """
        Create a math word quiz which involves a linear equation with two variables.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass

    @abstractmethod
    def create_quizzes(self, num_quizzes: int) -> Quizzes:
        """
        Create multiple quizzes.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass
