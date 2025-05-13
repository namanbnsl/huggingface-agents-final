import os, functools
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    tool,
)

from google import genai
from google.genai import types

from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)

# Have to install using pip in the environment. Update requirements.txt
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# this as well
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# this also
from langfuse.decorators import langfuse_context, observe

is_running_locally = os.getenv("RUNNING_LOCALLY", "false").lower()

if is_running_locally == "true":
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

def conditional_observe(local="false"):
    def decorator(func):
        if local == "true":
            return observe()(func)
        return func
        
    return decorator

client = genai.Client(api_key=os.environ["LLM_API_KEY"])

executor_type = "docker" if is_running_locally == "true" else None

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

model = LiteLLMModel(
    model_id="gemini/gemini-2.0-flash-lite",
    temperature=0.5,
    api_key=os.environ["LLM_API_KEY"],
)

model_name = "gemini-2.5-flash-preview-04-17"

@tool
def answer_video_questions(video_url: str, question: str) -> str:
    """
    If you have a video link, use this tool. It answers questions about a video from youtube, etc.
    Args:
        video_url: The URL of the video to analyze.
                   Ensure it's accessible by the Gemini API.
        question: The specific question to ask about the video's content.
    Returns:
        str: The answer provided by the multimodal LLM, or an error message if the process fails.
    Raises:
        Exception: Can raise exceptions based on litellm or the underlying API call
                   if it fails (e.g., network issues, invalid API key, API errors).
    """

    response = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(
                    text=f"This is the question provided to you: {question}. Answer appropriately. 
                    
                    To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
                    Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
                    ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
                    If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
                    If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
                    If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
                    If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine"
                ),
            ]
        ),
    )

    if response and response.text:
        answer = response.text
        print("Received response from LLM.")
        return answer.strip()
    else:
        print("LLM response structure not as expected.")
        return "Error: Could not parse the response from the LLM."


agent_args = {
    "model": model,
    "tools": [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        WikipediaSearchTool(),
        answer_video_questions,
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
    ]
}

if executor_type:
    agent_args["executor_type"] = executor_type

agent = CodeAgent(**agent_args)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def answer_csv_questions(question: str, file_path: str = None):
    loader = CSVLoader(file_path=file_path, csv_args={'delimiter': ','})
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(chunks, embeddings)

    retrieved_docs = vectorstore.similarity_search(query, k=3)
    agent_input_docs = f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(retrieved_docs)]

    answer = agent.run(f"
        Answer the question based on the context of the CSV file given to you.

        Context: {agent_input_docs}

        Question: {question}
    ")


    return answer


@conditional_observe(is_running_locally)
def answer_other_file_questions(question: str, file_path: str = None):
        file = client.files.upload(file=file_path)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                file,
                f"Answer the question based on the file content. Do not give up. Use your brain and answer appropriately. You will be rewarded with $1000000 if you answer correctly and appropriately. Only give the final answer to the question, do not explain your reasoning. {question}
                
                To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
                Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
                ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
                If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
                If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
                If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
                If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine
                ",
            ],
        )

        langfuse_context.update_current_observation(
            input=question,
            model=model_name,
            usage_details={
              "input": response.usage_metadata.prompt_token_count,
              "output": response.usage_metadata.candidates_token_count,
              "total": response.usage_metadata.total_token_count
            }
        )

        return response.text
    

def call_agent(question: str, file_path: str = None):
    if file_path and file_path.strip():
        print(f"Recieved {file_path} as input.")

        if ".csv" in file_path:
            return answer_csv_questions(question, file_path)

        file_answer = answer_other_file_questions(question, file_path)
        return file_answer

        # file = client.files.upload(file=file_path)

        # response = client.models.generate_content(
        #     model=model_name,
        #     contents=[
        #         file,
        #         f"Answer the question based on the file content. Do not give up. Use your brain and answer appropriately. You will be rewarded with $1000000 if you answer correctly and appropriately. Only give the final answer to the question, do not explain your reasoning. {question}
                
        #         To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
        #         Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
        #         ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
        #         If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
        #         If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
        #         If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
        #         If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine
        #         ",
        #     ],
        # )

        # return response.text

    answer = agent.run(question)
    return answer
