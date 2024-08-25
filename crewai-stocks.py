# Import's libs
import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# Yahoo Finance Tool
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} from the last year about a specific company stock from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)

# import Openai llm - gpt
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model = "gpt-3.5-turbo")

stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You're a highly experienced in analyzing the price of an specific stock and make predictions about its future price""",
    verbose=True,
    llm= llm,
    max_iter= 5,
    memory= True,
    allow_delegation= False,
    tools=[yahoo_finance_tool]
)

getStockPrice = Task(
    description="Analyse the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output="""Specify the current trend stock price - up, down or sideways.
    eg. stock='APPL, price UP'
    """,
    agent= stockPriceAnalyst
)

# Import tool of search
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticket} company. 
    Specify the current trend - up, down or sideways with the news context. 
    For each request stock asset, specify a number between 0 and 100 is extreme greed
    """,
    backstory="""You're a highly experienced in analyzing the market trends and news and have tracked assets for more then 10 years.
    You're also master level analyts in the traditional and have deep understanding of human psychology.
    You understanding news, theirs titles and information, but you look at those with a health dose of skepticism.
    """,
    verbose=True,
    llm= llm,
    max_iter= 10,
    memory= True,
    allow_delegation= False,
    tools=[search_tool]
)

get_news = Task(
    description="""Take the stock and always include BTC to it (if not request)
    Use the search tool to search each one individualy.

    The current date is {datetime}.

    Compose the results into a helpfull resport
    """,
    expected_output="""A summary of the overeal market and one setence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent= newsAnalyst
)

stockAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyse the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend""",
    backstory="""You're widely accepted as the best stock analyst in the market. You understanding complex concepts and create compelling 
    stories and narratives that resonate with wider audiences. 

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
    You're able to hold multiple opinions when analyzing anything.
    """,
    verbose=True,
    llm= llm,
    max_iter= 5,
    memory= True,
    allow_delegation= True,
    tools=[search_tool]
)

writeAnalyses = Task(
    description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
    """,
    expected_output= """"An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:

    - 3 bullets executive summary 
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fead/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.
    """,
    agent = stockAnalystWrite,
    context = [getStockPrice, get_news]
)

crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, get_news, writeAnalyses],
    verbose = True,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

results= crew.kickoff(inputs={'ticket': 'AAPL', 'datetime': '2024-08-25'})

results['final_output']

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of your research:")
        st.write(results['final_output'])