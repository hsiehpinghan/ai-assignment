from langchain_core.runnables.base import Runnable
from langchain_core.runnables.utils import Input
from langchain_core.runnables.utils import Output
from langchain_core.runnables.utils import Optional
from langchain_core.runnables.config import RunnableConfig

class ErrorHandler(Runnable):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return input["output_example"]
