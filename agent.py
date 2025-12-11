#!/usr/bin/env python
# coding: utf-8

# 
# # Build a Tool-Use ReAct Agentic AI System with LangGraph

# ## Problem Statement
# 
# In this notebook, we will be building an Agentic AI system — HealthBuddy — using LangGraph with built-in utility functions. This system will act as an intelligent assistant that can:
# 
# - Understand user queries about health
# - Research using web and arXiv tools
# - Recommend appropriate doctors based on user symptoms
# 
# We will follow the **ReAct principle** (Reason + Act) to let the agent think, use tools when needed, observe results, and provide a well-formed final answer. Instead of relying on built-in LLM knowledge alone, the agent will use **external tools** to gather accurate and helpful information before responding.
# 
# 
# ---
# 
# ### Objective:
# 
# Our goals in this notebook are:
# 
# - **Build a ReAct-based Tool-Use Agent using LangGraph with built-in utility functions like `create_react_agent()`**
# - Equip the agent with multiple tools (web search, arXiv search, doctor recommendation)
# - Handle user queries end-to-end — from interpreting intent to delivering a useful, cited, and grounded response
# - Simulate a multi-step reasoning and tool-using workflow
# 
# By the end of this notebook, we will have a working agent that can research a health query, gather relevant information, and offer advice including suggesting a doctor — all through structured decision-making and tool usage.
# 
# ### Agent Architecture:
# 
# The following figure shows the agent architecture including all the components and the overall workflow
# 
# ![](https://i.imgur.com/efDt1tq.png)
# 

# In[1]:


#get_ipython().system('pip install langgraph==1.0.2 langchain==1.0.5 langchain-openai==1.0.2 langchain-community==0.4.1 arxiv==2.3.0 pymupdf==1.26.6')


# ## Load Necessary Dependencies

# In[2]:


import json
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from IPython.display import display, Image, Markdown


# ## Setup Authentication and LLM Client
# 
# Here we authenticate and connect to necessary LLM using OpenAI Authentication

# In[3]:


import os
import getpass
from dotenv import load_dotenv, find_dotenv

load_dotenv('/Users/kbikal/Documents/GenAI/.env')

# OpenAI API Key (for chat & embeddings)
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key (https://platform.openai.com/account/api-keys):\n")
    
# Tavily API Key (for web search)
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key (https://app.tavily.com/home):\n")


# In[4]:


llm_client = ChatOpenAI(model_name="gpt-5-nano")


# In[5]:


llm_client.invoke("Explain what is AI in 1 line").content


# ## Preparing Database for Doctor Recommendations Tool
# 
# In this section, we will create a small, in-memory database that contains information about doctors. This data will be used by our **Doctor Recommendation Tool**, which will help users find the right doctor based on their health query or symptoms.
# 
# The database includes a list of doctors along with their:
# - Name
# - Specialization (e.g., Dermatology, Pediatrics, Cardiology)
# - Location
# - Availability
# - Contact information
# 
# We are using a simple Python list of dictionaries to store the doctor data. This keeps it easy to understand and modify. In a real-world application, this would typically be replaced by a backend database like PostgreSQL, MongoDB, or an external API.
# 
# We will build a tool later on to use this data to recommend doctors based on the user's needs — for example, suggesting a pediatrician for a child’s fever or a cardiologist for chest pain.

# In[6]:


# loading our doctors dataset
doctors_db = [
    {"name": "Dr. Janet Dyne", "specialization": "Endocrinology (Diabetes Care)", "available_timings": "10:00 AM - 1:00 PM", "location": "City Health Clinic", "contact": "janet.dyne@healthclinic.com"},
    {"name": "Dr. Don Blake ", "specialization": "Cardiology (Heart Specialist)", "available_timings": "2:00 PM - 5:00 PM", "location": "Metro Cardiac Center", "contact": "don.blake@metrocardiac.com"},
    {"name": "Dr. Susan D'Souza", "specialization": "Oncology (Cancer Care)", "available_timings": "11:00 AM - 2:00 PM", "location": "Hope Cancer Institute", "contact": "susan.dsouza@hopecancer.org"},
    {"name": "Dr. Matt Murdock", "specialization": "Psychiatry (Mental Health)", "available_timings": "4:00 PM - 7:00 PM", "location": "Mind Care Center", "contact": "matt.murdock@mindcare.com"},
    {"name": "Dr. Dinah Lance", "specialization": "General Physician", "available_timings": "9:00 AM - 12:00 PM", "location": "Downtown Medical Center", "contact": "dinah.lance@downtownmed.com"}
]


# ## Create Tools for AI Agent
# 
# In this section, we will define the tools that our AI Agent will use to perform specific tasks.
# 
# LangChain makes it easy to create and register tools using the `Tool` class. A tool includes:
# - A name and description
# - The python function to be called
# - An input schema that tells the model what arguments it can use
# 
# When tools are defined properly, they help the model solve more complex problems by letting it take actions and use external data. This makes the system more useful and reliable.
# 
# ### 🧪 Example
# ```python
# from langchain.tools import tool
# 
# @tool
# def search_web(query: str) -> str:
#     """Get live information for user queries from the web."""
#     # assuming we have a google_search function implemented to search on google
#     return google_search(query)
# ```
# 
# These tools will allow the agent to retrieve information from our preloaded vector databases (web search and PubMed), as well as recommend doctors from our in-memory doctor database.
# 
# The goal is to modularize the logic for different types of tasks into reusable components that can be invoked by the LLM when needed. These include:
# 
# - A **Web Search Tool** that queries the web to get relevant information through web search
# - A **ArXiv Search Tool** that retrieves information from top research papers from arxiv.org relevant to the query  
# - A **Doctor Recommendation Tool** that finds suitable doctors based on user symptoms or needs
# 
# This tool-based setup is essential for enabling agentic behavior, where the LLM reasons through a problem, decides which action to take, and requests to call the right tools to gather more information or perform a task.
# 

# In[7]:


from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.retrievers import ArxivRetriever

# Tool for web search on general health topics
# Tavily Web Search
tavily_search = TavilySearchAPIWrapper()

@tool
def search_web(query: str) -> list:
    """
    Search the web for general or up-to-date information on healthcare topics.

    Inputs:
    - query (str): The search query string. Should describe the healthcare topic or information you want to find.

    Outputs:
    - list: A list of up to 3 formatted strings, each containing:
        - Title of the search result
        - Content extracted from the page
        - Source URL
    """
    results = tavily_search.raw_results(query=query, max_results=3, search_depth='advanced',
                                        include_answer=False, include_raw_content=True)
    docs = results['results']
    docs = [doc for doc in docs if doc.get("raw_content") is not None]
    docs = ['## Title\n'+doc['title']+'\n\n'+'## Content\n'+doc['raw_content']+'\n\n'+'##Source\n'+doc['url'] for doc in docs]
    return docs


# Tool for arXiv Search
# arxiv search retriever
arxiv_retriever = ArxivRetriever(
    top_k_results=3,
    get_full_documents=True,
    doc_content_chars_max=20000
)

@tool
def search_arxiv(query: str) -> list:
    """
    Search arXiv for relevant scientific research papers and articles.

    Inputs:
    - query (str): The research topic or keywords to search for on arXiv.

    Outputs:
    - list: A list of up to 3 formatted strings, each containing:
        - Title of the paper
        - Summary of the paper
        - Full content (truncated to maximum allowed characters)
      Returns ["No articles found for the given query."] if no matches are found.
    """
    try:
        results = arxiv_retriever.invoke(query)
        if results:
            articles = ['## Title\n'+doc.metadata['Title']+'\n\n'+'## Summary\n'+doc.metadata['Summary']+'\n\n'+'##Content\n'+doc.page_content for doc in results]
            return articles
        else:
            return ["No articles found for the given query."]
    except Exception as e:
        return [f"Error fetching arXiv articles: {str(e)}"]


@tool
def recommend_doctor(query: str) -> dict:
    """
    Recommend the most suitable doctor based on the user's symptoms or health-related query.

    Inputs:
    - query (str): A description of the patient's symptoms or healthcare needs.

    Outputs:
    - dict: A dictionary containing:
        - "recommended_doctor": JSON-formatted details of the selected doctor from the `doctors_db`.
          If the most suitable match cannot be determined, defaults to recommending the General Physician.
    """
    doctors_list = str(doctors_db)

    prompt = f"""You are an assistant helping recommend a doctor based on patient's health issues.

                 Here is the list of available doctors:
                {doctors_list}

                Given the user's query: "{query}"

                Choose the most suitable doctor from the list. Only pick one doctor.
                Return only the selected doctor's information in JSON format (no markdown).
                If not sure, recommend the General Physician.
              """
    response = llm_client.invoke(prompt)
    return {"recommended_doctor": response}


# In[8]:


print(recommend_doctor.invoke('need a doc for diabetes')['recommended_doctor'].content)


# ## Define Tools to be Used by the Agent
# 
# In this section, we define the tools that our AI agent will use when reasoning through user queries. These include:
# 
# - `search_web`: To fetch general health-related information from the internet.
# - `search_arxiv`: To retrieve scientific research papers from arXiv.org.
# - `recommend_doctor`: To recommend a suitable doctor based on the user's symptoms.
# 
# These tools are already defined and registered using the `@tool` decorator. Here, we simply collect them into a list to be fed into the agent function later on.
# 

# In[9]:


# List of all tools that the LLM should be aware of
# These tools were defined earlier using the @tool decorator
tools = [search_web, search_arxiv, recommend_doctor]


# ## Define Agent Instructions Prompt
# 
# To guide the LLM-based agent, we provide a custom system prompt that sets the role and behavior of the assistant.
# 
# The prompt clearly instructs the agent to:
# - Act as a helpful healthcare assistant
# - Reason on input queries and do the following:
#   - Research the query using the most relevant tools (web, pubmed)
#   - Recommend a doctor only if appropriate
# 
# This prompt plays a critical role in shaping how the agent reasons, decides when to call tools, and how to construct final responses. It supports ReAct-style behavior where the model reflects, takes actions (via tools), and continues reasoning.

# In[10]:


# Instruction prompt for the overall Agent
AGENT_PROMPT_TXT = r"""You are an agent designed to act as an expert in researching on medical symptoms
and also recommend relevant doctors for booking appointments.
Also remember the current year is 2025 and use the same for all search queries when no specific dates are mentioned.

Given an input user query call relevant tools and give the most appropriate response.
Follow some of these guidelines to help you make more informed decisions:
  - If the user's query specifies recommending a doctor only then recommend an appropriate doctor
  - If the user is researching on detailed and specific aspects around symptoms, treatments and other aspects related to healthcare
  use both search_web and search_arxiv tools to get comprehensive information and then give a well-structured response
  - If the user is just looking for general information around healthcare then web search is enough
  - Use search_arxiv tool only if the query is related to information which might be found in research papers
  - Response should include cited source links and \ or arXiv Article Title, Publication Dates if available
  - If recommending doctors then use the recommend_doctor and show detailed information in a nice structured way and recommend them to book an appointment via email
  - Politely decline answering any queries not related to medical or healthcare information
"""

AGENT_SYS_PROMPT = SystemMessage(content=AGENT_PROMPT_TXT)


# ## Create our Tool-Use AI Agent
# 
# Using the tools, prompt, and LLM setup, we now create the actual **Tool-Use Agent** using LangGraph's built-in `create_react_agent()` utility function.
# 
# This creates a simple ReAct-style agent capable of:
# - Receiving a user query
# - Reasoning on next steps
# - Calling the appropriate tool(s) if needed
# - Observing tool outputs
# - Returning a final answer
# 
# This step brings together everything we’ve built so far: tools, instructions, and reasoning logic into a functioning agentic loop.

# In[11]:


# Create our agent using tools, LLM and instruction prompt
healthbuddy_agent = create_react_agent(model=llm_client,
                                       tools=tools,
                                       prompt=AGENT_SYS_PROMPT)


# You can update this and use the `create_agent` function from LangChain agents only if you are using version > 1.0, in production if you are using version < 1.0 then you can continue using the above (recommended especially as LangChain 1.0 just came out and expect them to break more stuff!)

# In[12]:


# ONLY if you want to use langchain and langgraph > 1.0
from langchain.agents import create_agent

healthbuddy_agent = create_agent(model=llm_client,
                                 tools=tools,
                                 system_prompt=AGENT_PROMPT_TXT) # just send prompt text directly here - do NOT need a langchain System Message

def run_agent(prompt: str):
    """
    Simple wrapper function to run the HealthBuddy agent
    and return only the final response text.
    """

    try:
        # Run the agent (no streaming)
        events = healthbuddy_agent.invoke(
            {"messages": [HumanMessage(content=prompt)]}
        )

        # Extract the final LLM message
        final_msg = events["messages"][-1].content

        return final_msg

    except Exception as e:
        return f"Error running agent: {str(e)}"
    
# ## View our Agent Flow
# 

# In[13]:


display(Image(healthbuddy_agent.get_graph().draw_mermaid_png()))


# ## Build Utility Function to Stream Agent Results
# 
# We define a helper function to stream the step-by-step output of the agent. This makes it easier to trace:
# - What the agent doing in each step
# - Which tool it decides to call
# - What response it gets from that tool
# - How it forms the final reply
# 
# Streaming output is helpful when evaluating multi-step reasoning and debugging tool use in real time.

# In[14]:


# get agent streaming utils
get_ipython().system('gdown 1dSyjcjlFoZpYEqv4P9Oi0-kU2gIoolMB')


# In[15]:


from agent_utils import format_message

# Utility function to call the agent and stream its step-by-step reasoning
def call_agent(agent, query, verbose=False):

    # Stream the agent's execution with the given query
    for event in agent.stream(
        {"messages": [HumanMessage(content=query)]}, # input prompt
        stream_mode='values'  # Stream output as intermediate values
    ):
        # If verbose is enabled, print each intermediate message step-by-step
        if verbose:
            format_message(event["messages"][-1])

    # Display the final response from the agent as Markdown
    print('\n\nFinal Response:\n')
    display(Markdown(event["messages"][-1].content))

    # Return the overall event messages for optional downstream use
    return event["messages"]


# ## Test out our Agent!
# 
# In this final section, we run a complete test of our Tool-Use AI Agent by passing it a sample health-related query.
# 
# We observe how the agent:
# - Interprets the query
# - Decides which tool(s) to use (if any)
# - Executes tool calls
# - Streams the intermediate steps and final output

# In[16]:


# Example usage
query = "what are the latest methods for diabetes management and recommend a doctor please"
result = call_agent(healthbuddy_agent, query, verbose=True)


# In[17]:


# inspect all agent events separately
for event in result:
    format_message(event)


# In[18]:


# Example usage
query = "I am having panic attacks, what could I do? get detailed comprehensive information please"
result = call_agent(healthbuddy_agent, query, verbose=True)


# In[19]:


# Example usage without printing detailed log messages
query = "I am having panic attacks, please recommend a right doctor"
result = call_agent(healthbuddy_agent, query, verbose=False)


# In[20]:


queries = [
    'Patient 001: Reported panic attacks - please assign doctor',
    'Patient 003: Diabetes checkup - please assign doctor',
    'Patien 004: Has the flu - please assign doctor'
]


# In[21]:


results = [call_agent(healthbuddy_agent, query, verbose=False) for query in queries]


# In[22]:


import pandas as pd
pd.set_option('display.max_colwidth', None)
res_df = pd.DataFrame({'query': queries,
                       'response': [item[-1].content for item in results]})
res_df


# In[ ]:




