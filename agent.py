import os
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

client = genai.Client(api_key=os.environ["LLM_API_KEY"])

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
                    text=f"This is the question provided to you: {question}. Answer appropriately. In cases where there is a number, provide the number in the answer."
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


agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        WikipediaSearchTool(),
        answer_video_questions,
    ],
    model=model,
)


def call_agent(question: str, file_path: str = None):
    if file_path and file_path.strip():
        print(f"Recieved {file_path} as input.")

        file = client.files.upload(file=file_path)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                file,
                f"Answer the question based on the file content. Do not give up. Use your brain and answer appropriately. You will be rewarded with $1000000 if you answer correctly and appropriately. Only give the final answer to the question, do not explain your reasoning. {question}",
            ],
        )

        return response.text

    answer = agent.run(question)
    return answer
