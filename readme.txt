# init poetry
cd C:\Users\thank\git\ai-assignment
poetry init
    Package name [ai-assignment]:  
    Version [0.1.0]:  
    Description []:  
    Author [hsiehpinghan <thank.hsiehpinghan@gmail.com>, n to skip]:  
    License []:  
    Compatible Python versions [^3.12]:  3.11.10
    Would you like to define your main dependencies interactively? (yes/no) [no] 
    Would you like to define your development dependencies interactively? (yes/no) [no]
    Do you confirm generation? (yes/no) [yes] 
poetry config --list
poetry config virtualenvs.in-project true
mkdir C:\Users\thank\git\ai-assignment\.venv
python -m venv C:\Users\thank\git\ai-assignment\.venv
poetry env use C:\Users\thank\miniconda3\envs\python3.11\python.exe
poetry env info
poetry add langchain==0.1.0
poetry add langchain-community==0.0.10
poetry add langchain-core==0.1.11
poetry add langchain-experimental==0.0.49
poetry add langchain-openai==0.0.2
poetry add openai==1.7.0
poetry add pydantic==2.3.0
poetry add python-dotenv==1.0.1
poetry install