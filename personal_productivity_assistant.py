import os
import dateparser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tasks = []
reminders = []

@tool
def add_task(title: str, deadline: str) -> str:
    """Adds a new task with a deadline."""
    try:
        parsed_deadline = dateparser.parse(deadline)
        task = {
            "title": title,
            "deadline": parsed_deadline.date()
        }
        tasks.append(task)
        return f"Task '{title}' added with deadline {task['deadline']}"
    except Exception as e:
        return f"Error adding task: {e}"

@tool
def set_reminder(task_title: str, reminder_time: str, priority: str = "medium") -> str:
    """Sets a reminder for a task at a specific date and time with a priority."""
    try:
        parsed_datetime = dateparser.parse(reminder_time)
        reminder = {
            "task_title": task_title,
            "reminder_time": parsed_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "priority": priority if priority else "medium"
        }
        reminders.append(reminder)
        return f"Reminder for '{task_title}' set at {reminder['reminder_time']} with priority {priority}"
    except Exception as e:
        return f"Error setting reminder: {e}"

@tool
def get_query(query_type: str) -> str:
    """Retrieves tasks or reminders based on the query type."""
    if query_type == "tasks":
        return f"Current tasks: {tasks}"
    elif query_type == "reminders":
        return f"Current reminders: {reminders}"
    else:
        return "Invalid query type. Use 'tasks' or 'reminders'"

tools = [add_task, set_reminder, get_query]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

# Create the system prompt
system_prompt = """
You are a helpful personal productivity assistant.
You can help users manage tasks and set reminders.
If the user request is unclear, ask follow-up questions to clarify (e.g., “I need help organizing my day” → “Would you like me to create a task, set a reminder, or view existing ones?”).
If the request is very ambiguous or unrelated to tasks and reminders, respond with: "I can help with creating a task or setting up a reminder. Let me know what you’d like to do."
Always respond in a friendly and helpful manner.
"""



# Create the prompt template with agent_scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent using LCEL
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create an LCEL chain
chain = agent_executor

def chat_with_agent():
    print("Welcome to the Personal Productivity Assistant!")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Pass the input to the agent using LCEL
        response = chain.invoke({"input": user_input})
        print(f"Agent: {response['output']}")

if __name__ == "__main__":
    chat_with_agent()