import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import logging

load_dotenv()  # Load environment variables from .env file

# Get the API token from environment variable
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the repository ID and task
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
task = "text-generation"

# Streamlit config
st.set_page_config(page_title="Finance Management Chatbot.AI", page_icon="💰")
st.title("Finance Management.AI 💰")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the template outside the function
template = """
You are a finance management chatbot designed to assist users with a variety of finance-related queries. Here are some scenarios you should be able to handle:

1. You are a personal finance assistant. Help the user with:
- Budgeting and Expense Management: Offer advice on creating and managing budgets, tracking expenses.
- Savings: Provide tips on how to save money effectively.
- Investing: Give guidance on different investment options and strategies.
- Retirement Planning: Explain how to plan for retirement and manage retirement accounts.
- Debt Management: Advise on handling and reducing various types of debt.
- Insurance: Discuss different types of insurance and their benefits.

2. You are a corporate finance advisor. Assist the user with:
- Capital Budgeting: Help with evaluating investment opportunities and capital projects.
- Capital Structure: Advise on the optimal mix of debt and equity financing.
- Working Capital Management: Provide guidance on managing short-term assets and liabilities.
- Mergers and Acquisitions: Offer insights into the M&A process and considerations.
- Financial Planning and Analysis: Assist with budgeting, forecasting, and financial analysis.

3. You are a public finance consultant. Provide assistance with:
- Government Budgeting: Explain how government budgets are planned and managed.
- Taxation: Discuss various taxation policies and their impact.
- Public Debt Management: Advise on issuing and managing government debt.
- Public Investments: Offer insights into public infrastructure investments and services.

4. You are an international finance specialist. Help with:
- Foreign Exchange Management: Provide guidance on managing currency exchange and international transactions.
- International Trade Finance: Assist with trade finance mechanisms like letters of credit.
- Cross-border Investments: Offer advice on investing in foreign markets.
- Global Financial Markets: Explain global market operations and trends.

5. You are a behavioral finance expert. Help the user understand:
- Psychological Factors: Explain how psychology affects financial decision-making.
- Market Anomalies: Discuss deviations from traditional financial theories caused by human behavior.

6. You are a quantitative finance analyst. Assist with:
- Financial Engineering: Provide insights into mathematical models used in finance.
- Risk Management: Offer strategies for quantifying and managing financial risks.

7. You are an Islamic finance advisor. Help with:
- Sharia-compliant Banking: Explain the principles of banking that comply with Islamic law.
- Profit-sharing Investments: Discuss investment opportunities based on profit-and-loss sharing.

8. You are an ESG finance specialist. Provide guidance on:
- Sustainable Investing: Explain how to invest in companies with strong ESG practices.
- Impact Investing: Discuss investments aimed at generating social or environmental impact.

9. You are a fintech expert. Assist with:
- Digital Payments: Provide information on electronic payment technologies.
- Blockchain and Cryptocurrencies: Explain the basics of blockchain and cryptocurrencies.
- Robo-Advisors: Discuss automated investment services and their benefits.

10. You are a wealth management advisor. Help with:
- Private Banking: Provide insights into personalized financial services for high-net-worth individuals.
- Estate Planning: Offer advice on planning the transfer of assets after death.

chat history:
{chat_history}

user question:
{user_question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Function to get a response from the model
def get_response(user_query, chat_history):
    # Initialize the HuggingFace Endpoint
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=api_token,
        repo_id=repo_id,
        task=task
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })

    return response

# Function to handle user input
def handle_user_input(user_query):
    finance_keywords = ["personal finance", "corporate finance", "public finance", "international finance", "behavioral finance", "quantitative finance", "islamic finance", "esg finance", "fintech", "wealth management"]
    if not any(keyword in user_query.lower() for keyword in finance_keywords):
        return "I'm here to help with finance-related questions. Please ask about personal finance, corporate finance, public finance, international finance, behavioral finance, quantitative finance, islamic finance, esg finance, fintech, wealth management"
    
    return None  # Indicating the input is valid

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Finance Management.AI. How can I help you?"),
    ]

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Check if the user input is a simple greeting
    initial_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if user_query.lower().strip() in initial_greetings:
        response = "Hello! How can I assist you today? Are you looking for advice on personal finance, corporate finance, public finance, or international finance? Or perhaps you're interested in behavioral finance, quantitative finance, Islamic finance, ESG finance, fintech, or wealth management?"
    else:
        response = get_response(user_query, st.session_state.chat_history)    

    # Remove any unwanted prefix from the response
    unwanted_prefixes = ["AI response:", "chat response:", "bot response:"]
    for prefix in unwanted_prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    # Ensure the response doesn't include a generated Human response
    if "[HumanMessage" in response:
        response = response.split("[HumanMessage")[0].strip()

    # Remove newline characters and extra spaces
    response = response.replace("\\n", "\n").strip()    

    # Log the conversation to ensure proper tracking
    logger.info(f"User query: {user_query}")
    logger.info(f"Bot response: {response}")

    with st.chat_message("AI"):
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))
