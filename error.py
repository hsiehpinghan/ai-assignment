from langchain_core.runnables.base import Runnable
from langchain_core.runnables.utils import Input
from langchain_core.runnables.utils import Output
from langchain_core.runnables.utils import Optional
from langchain_core.runnables.config import RunnableConfig

class ErrorHandler(Runnable):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """
        If an error occurs, it returns "output_example".

        Args:
            input (Input): The input dictionary.
            config (Optional[RunnableConfig], optional): Optional runnable configuration. Defaults to None.

        Returns:
            Output: Returns "output_example" if an error occurs.
        """
        return input["output_example"]
