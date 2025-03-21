import datetime
import os
from datetime import datetime

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
        parsed_deadline = datetime.fromisoformat(deadline)
        task = {
            "title": title,
            "deadline": parsed_deadline.isoformat()
        }
        tasks.append(task)
        return f"Task '{title}' added with deadline {parsed_deadline.strftime('%A, %d %b %Y, %I:%M %p')}"
    except Exception as e:
        return f"Error adding task: {e}"

@tool
def set_reminder(task_title: str, reminder_time: str, priority: str = "medium") -> str:
    """Sets a reminder for a task at a specific date and time with a priority."""
    try:
        parsed_datetime = datetime.fromisoformat(reminder_time)
        reminder = {
            "task_title": task_title,
            "reminder_time": parsed_datetime.isoformat(),
            "priority": priority.lower()
        }
        reminders.append(reminder)
        return f"Reminder for '{task_title}' set at {parsed_datetime.strftime('%A, %d %b %Y, %I:%M %p')} with priority '{priority}'"
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

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

system_prompt = """
You are a helpful personal productivity assistant.
You can help users manage tasks and set reminders.
- If the user request is unclear, ask follow-up questions to clarify.
- If the request is ambiguous or unrelated to tasks and reminders, respond with:
    "I can help with creating a task or setting up a reminder. Let me know what you'd like to do."
- Process date and time in requests carefully. Always convert user's natural language dates to ISO format (YYYY-MM-DDTHH:MM:SS) before using tools.
- The current time is for your reference: {current_time}

IMPORTANT: Convert natural language dates to ISO format strings yourself.
When handling relative dates like "tomorrow", "next Thursday", or time references like "5 PM", convert them to proper ISO format dates based on the current time.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt.format(current_time=datetime.now().isoformat())),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Function to test with predefined questions
def test_questions():
    questions = [
        "Add a task to submit the project report by Monday.",
        "Create a task to buy groceries before 5 PM tomorrow.",
        "I need to finish the client presentation by Friday evening.",
        "Schedule a task for completing my assignment by next Wednesday.",
        "Can you remind me to renew my subscription before the end of the month?",
        "Remind me to call John at 3 PM with high priority.",
        "Set a reminder for my doctor's appointment at 10 AM on Saturday.",
        "I need a reminder to pay my electricity bill on the 5th at 9 AM.",
        "Remind me to send the weekly report every Monday at 8 AM, high priority.",
        "Please set a low-priority reminder to water the plants at 6 AM.",
        "What tasks do I have today?",
        "Show me my pending tasks for this week.",
        "What reminders are scheduled for tomorrow?",
        "Tell me all my high-priority tasks.",
        "Do I have any tasks due this Friday?",
        "Help me organize my work.",
        "I have a meeting.",
        "Remind me about my important things.",
        "I need to finish something soon.",
        "What do I need to do?"
    ]
    
    print("\n=== Running Test Questions ===\n")
    function_call_count = 0
    
    for i, question in enumerate(questions):
        print(f">> Test {i + 1}: {question}")
        response = agent_executor.invoke({"input": question})
        print(f"Agent: {response['output']}\n" + "-"*50)
        
        # Check if a function was called by inspecting the agent's response
        if any(tool.name in response['output'] for tool in tools):
            function_call_count += 1
    
    print(f"\n✅ Total function calls made: {function_call_count} / {len(questions)}")

def chat_with_agent():
    print("Welcome to the Personal Productivity Assistant!")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        response = agent_executor.invoke({"input": user_input})
        print(f"Agent: {response['output']}")

if __name__ == "__main__":
    mode = input("Type 'test' to run test cases or 'chat' to start chatting: ").strip().lower()
    if mode == "test":
        test_questions()
    else:
        chat_with_agent()