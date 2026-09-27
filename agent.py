import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

# Load environment variables (ensure OPENAI_API_KEY is in your .env file)
load_dotenv()

# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# Initialize the LLM (GPT-4o)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the LangGraph ReAct agent
agent = create_react_agent(llm, tools=[search_tool])

def main():
    print("Agent is ready! Ask a question or type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        # The agent requires a dictionary with a "messages" list
        inputs = {"messages": [("user", user_input)]}
        
        try:
            # Invoke the graph
            result = agent.invoke(inputs)
            # The final answer is the content of the last generated message
            print(f"\nAgent: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()