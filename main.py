from langchain_openai import ChatOpenAI
from langgraph import graph
from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import pprint
from langgraph.graph import END, StateGraph
import ssl; ssl._create_default_https_context = ssl._create_unverified_context

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key="")

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]



#defining nodes

def get_youtube_video_urls(state):
    print("---Fetching URls---")
    state_dict = state["keys"]
    question = state_dict["question"]
    top_k = state_dict["top_k"]
    
    tool = YouTubeSearchTool()
    if "docs" in state_dict:
        print("---GETTING NEW VIDEO URLS---", top_k+top_k)
        url_list = tool.run(question+","+str(top_k+top_k))
    else:
        url_list = tool.run(question+","+str(top_k))
    
    final_urls = []
    for u in url_list.split("'"):
        if "http" in u:
            final_urls.append(u)
    return {"keys": {"url_list":final_urls, "question":question, "top_k":top_k}}    

def get_video_text_from_urls(state):
    print("---Fetching Text from Video---")
    state_dict = state["keys"]
    question = state_dict["question"]
    top_k = state_dict["top_k"]
    final_urls = state_dict["url_list"]
    docs = []
    
    for url in final_urls:
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language=["en"],
                translation="en",
            )
            text = loader.load()
            if text and len(text) > 0:
                docs.append(text[0])
            else:
                print(f"---No transcript available for URL: {url}---")
        except Exception as e:
            print(f"---Error fetching transcript for URL: {url} - {e}---")
    
    return {"keys": {"docs": docs, "url_list": final_urls, "question": question, "top_k": top_k}}


def get_text_summary(state):
    print("---Creating Video Text Summary---")
    state_dict = state["keys"]
    question = state_dict["question"]
    final_urls = state_dict["url_list"]
    top_k = state_dict["top_k"]
    docs = state_dict["docs"]
    
    prompt_template = """Write a detailed summary highlighting key points to engage users of the following text:
    {text}
    Detailed Summary:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final detailed summary which should include the existing summary and only include "
        "relevant new key points which can improve the summary from the 'new context' provided. \n"
        "We have provided an 'existing summary' up to a certain point: Existing summary - \n {existing_answer}\n"
        "You have the opportunity to refine the existing summary, make sure to add only new points "
        "(only if needed) with some more context below. Don't remove key points from existing summary \n"
        "New Context - \n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the existing summary in English detailing all key points"
        "If the context isn't useful, return the Existing summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
         return_intermediate_steps=True,
        
        input_key="input_documents",
        output_key="output_text",
    )
    result = chain({"input_documents": docs}, return_only_outputs=True)
    
    return {"keys": {"generated_summary":result["output_text"], "generated_intermediate_summary":result["intermediate_steps"], "docs":docs, "url_list":final_urls, "question":question, "top_k":top_k}}



#defining edges

def decide_to_get_summary(state):
    """
    Determines whether to generate the summary, or get new video urls.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """
    print("---DECIDE TO GET SUMMARY---")
    
    state_dict = state["keys"]
    docs = state_dict["docs"]

    if len(docs)>=2:
       
        print("---DECISION: GET TEXT SUMMARY---")
        return "get_text_summary"
    else:
        
        print("---DECISION: GET NEW VIDEO URLS---")
        return "get_youtube_video_urls"
    

#building graph


workflow = StateGraph(State)

#nodes
workflow.add_node("get_youtube_video_urls", get_youtube_video_urls)
workflow.add_node("get_video_text_from_urls", get_video_text_from_urls)
workflow.add_node("get_text_summary", get_text_summary)

#graph
workflow.set_entry_point("get_youtube_video_urls")
workflow.add_edge("get_youtube_video_urls", "get_video_text_from_urls")
workflow.add_conditional_edges(
    "get_video_text_from_urls",
    decide_to_get_summary,
    {
        "get_text_summary": "get_text_summary",
        "get_youtube_video_urls": "get_youtube_video_urls",
    },
)
workflow.add_edge("get_text_summary", END)
app = workflow.compile()

inputs = {"keys": {"question": "What is retrieval augmented generation","top_k":3}}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
    pprint.pprint("\n---\n")

#final response
pprint.pprint(value['keys']['generated_summary'])
