from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph # type
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import  HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import chainlit as cl
from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
import os
from fastapi import FastAPI, HTTPException
import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# Setup the RAG model and FAISS VectorStore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

try:
    loader = TextLoader("hospitals.txt")
    index_creator = VectorstoreIndexCreator(
        embedding=embedding, 
        vectorstore_cls=FAISS,
        text_splitter=text_splitter
    )
    index = index_creator.from_loaders([loader])
except Exception as e:
    print("Error while loading or indexing the document:", e)
    index = None  # Set to None if there's an issue

# Define the RAG query tool for nutritionist data
def rag_query_tool(user_input: str) -> str:
    """
    Query the RAG model to retrieve hospitals and doctors information.

    Args:
        user_input: The query string from the user.

    Returns:
        A string response with nutritionist details or an error message.
    """
    if not index:
        return "hospitals data index is not available."
    try:
        response = index.query(user_input, llm=llm)

        if response:
            # Assuming response is structured, return it formatted.
            return response
        else:
            return "No relevant hospitals information found for your query."
    except Exception as e:
        return f"Error while querying hospitals data: {e}"



# Define the database model for appointments
class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doctor: str
    day: str
    time: str
    specialization: str

# Create a SQLite database
engine = create_engine(os.getenv('DB_URI'))
SQLModel.metadata.create_all(engine)

# The book_appointment function
def book_appointment(doctor: str, day: str, time: str, specialization: str) -> str:
    """Books an appointment and saves it in the database.
    
    Args:
        doctor: The name of the doctor with whom the appointment is booked.
        specialization: The specialization of the doctor.
        date: The day of the appointment (any week day from Monday to sunday).
        time: The time of the appointment (HH:MM).
    
    Returns:
        Confirmation message with appointment details.
    """
    try:
        # Use correct field names
        appointment = Appointment(doctor=doctor, day=day, time=time, specialization=specialization)
        with Session(engine) as session:
            session.add(appointment)
            session.commit()
        return f"Appointment booked successfully with Dr. {doctor} (Specialization: {specialization}) on {day} at {time}."
    except Exception as e:
        return f"Failed to book appointment: {str(e)}"

def read_all_appointments() -> str:
    """Fetches and reads all appointments from the database.
    
    Returns:
        A string containing all appointment details, or a message if no appointments are found.
    """
    try:
        with Session(engine) as session:
            appointments = session.query(Appointment).all()

            if appointments:
                appointment_details = ""
                for appointment in appointments:
                    appointment_details += (f"Appointment with Dr. {appointment.doctor} "
                                            f"(Specialization: {appointment.specialization}) on "
                                            f"{appointment.day} at {appointment.time}.\n")
                return appointment_details
            else:
                return "No appointments found in the database."
    except Exception as e:
        return f"Failed to retrieve appointments: {str(e)}"


def cancel_appointment(doctor: str, day: str, time: str) -> str:
    """Cancels an appointment by searching with doctor, day, or time.
    
    Args:
        doctor: The name of the doctor (optional).
        day: The day of the appointment (optional).
        time: The time of the appointment (optional).
    
    Returns:
        Confirmation message or an error message.
    """
    try:
        with Session(engine) as session:
            query = session.query(Appointment)
            if doctor:
                query = query.filter(Appointment.doctor == doctor)
            if day:
                query = query.filter(Appointment.day == day)
            if time:
                query = query.filter(Appointment.time == time)

            appointment = query.first()

            if appointment:
                session.delete(appointment)
                session.commit()
                return f"Appointment with Dr. {appointment.doctor} on {appointment.day} at {appointment.time} has been canceled."
            else:
                return "Appointment not found with the provided details."
    except Exception as e:
        return f"Failed to cancel appointment: {str(e)}"

def update_appointment(doctor: str, day: str, time: str, 
                                   new_doctor: str, new_day: str, new_time: str, 
                                   new_specialization: str) -> str:
    """Updates an appointment by searching with doctor, day, or time.
    Deletes the old appointment and creates a new one.
    
    Args:
        doctor: The current doctor's name (optional).
        day: The current day of the appointment (optional).
        time: The current time of the appointment (optional).
        new_doctor: The new doctor's name (optional).
        new_day: The new day of the appointment (optional).
        new_time: The new time of the appointment (optional).
        new_specialization: The new specialization of the doctor (optional).
    
    Returns:
        Confirmation message or an error message.
    """
    try:
        with Session(engine) as session:
            # Search for the existing appointment
            query = session.query(Appointment)
            if doctor:
                query = query.filter(Appointment.doctor == doctor)
            if day:
                query = query.filter(Appointment.day == day)
            if time:
                query = query.filter(Appointment.time == time)

            appointment = query.first()

            if appointment:
                # Delete the old appointment
                session.delete(appointment)
                session.commit()

                # Create a new appointment with the updated details
                new_appointment = Appointment(
                    doctor=new_doctor or appointment.doctor,
                    day=new_day or appointment.day,
                    time=new_time or appointment.time,
                    specialization=new_specialization or appointment.specialization
                )
                session.add(new_appointment)
                session.commit()

                return f"Appointment successfully updated to Dr. {new_doctor or appointment.doctor} (Specialization: {new_specialization or appointment.specialization}) on {new_day or appointment.day} at {new_time or appointment.time}."
            else:
                return "Appointment not found with the provided details."
    except Exception as e:
        return f"Failed to update appointment: {str(e)}"


def send_appointment_email(to_email: str, appointment_date: str, appointment_time: str) -> None:
    """
    Sends an appointment confirmation email to the user.

    Args:
        to_email (str): Recipient's email address.
        appointment_date (str): Date of the appointment.
        appointment_time (str): Time of the appointment.
    """
    # Email account credentials
    sender_email = "masfanasrullahansari123@gmail.com"
    sender_password = "wwrv dyjw xrca qlwh"

    # Email content
    subject = "Appointment Confirmation"
    body = (
        f"Dear User,\n\n"
        f"Your appointment has been successfully booked for {appointment_date} at {appointment_time}.\n\n"
        f"Thank you!"
    )

    # Create the email
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = to_email
    message['Subject'] = subject

    # Attach the email body
    message.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)
        server.send_message(message)
        print(f"Appointment confirmation email sent to {to_email}.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")
    finally:
        server.quit()

search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

loader1 = WebBaseLoader("https://www.mayoclinic.org/diseases-conditions")
# loader2 = WebBaseLoader("https://www.msdmanuals.com/home")
# loader3 = WebBaseLoader("https://www.eatingwell.com/category/4305/weight-loss-meal-plans/")
docs1 = loader1.load()
# docs2 = loader2.load()
# docs3 = loader3.load()
# combined_docs = docs1 + docs2 + docs3
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs1)
vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "mayoclinic_search",
    "Search for information about disease and patient condition. You must use this tool!",
)


tools = [search, retriever_tool, book_appointment, update_appointment, cancel_appointment, read_all_appointments, rag_query_tool]


llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content='''You are a knowledgeable and supportive assistant specializing in hospital and healthcare services. 
Your key responsibilities include providing information about hospitals, doctors, specializations, and assisting with appointment bookings (or video conceltation) for healthcare professionals.

### **Hospital Information Access via rag_query_tool (from hospital.txt file):
- You can retrieve detailed information about hospitals using the **rag_query_tool**. This tool allows you to fetch hospital-related data, such as:
  - **Hospital Name**
  - **Address**
  - **Contact Number**
  - **Website Link**
- If the user asks for hospital details or list of hospital then fetch them from rag tool (hospital file), you will query this information using the **rag_query_tool** to return relevant data about hospitals from a pre-configured dataset. This includes the hospital’s name, contact info, location, and website, ensuring that users get accurate and up-to-date information.
- If user specify their disease then find doctor according to that disease then search disease relate dortors and provide them their detail and ask fo book appointment (or video conceltation)
- If user ask about specific doctor (with its name or speialization) then provide the details and If user ask about that doctor is from which hospital then you haeve to mention that hospital.
- If I ask about the dooctor for perticular issue then you give me the doctors related to that problem like I want to have a doctor for my skin issues then you provide me a list of dermatologists.
- If User ask about the availables rooms then to show the details of that specific unit.

### **Book Appointment Assistant**:
- If the user requests an appointment or conceltation, follow these steps:
  1. Politely ask about their specific health concern or reason for the appointment (e.g., Dermatologist, dentist, cardiologist etc).
  2. Use the **rag_query_tool** to fetch a list of available doctors based on the user's requirements, showing their names, specializations, and available days and times.
  3. Present the user with the list of available doctors and their schedules, and ask them to select a preferred doctor, day, and time.
  4. Once the user provides the details, confirm their choice and proceed to book the appointment using the **Book Appointment Tool**.
  5. Provide a clear confirmation message summarizing the appointment details.

### **Appointment Management Functions**:
- **Book Appointment**:
    - This function allows users to book an appointment or online consultation with a doctor based on their specialization, preferred day, and time.
    - Once the appointment is successfully booked, the system will confirm the booking and provide the details.

- **Cancel Appointment**:
    - If the user needs to cancel an appointment, this function searches for the appointment by details such as doctor name, day, and time, and deletes it.
    - A confirmation message will be provided once the appointment is successfully canceled.

- **Update Appointment**:
    - When the user requests to update an appointment, the system will first delete the existing appointment and then create a new one with the updated details.
    - It ensures the user’s needs are met by reflecting the correct changes to the doctor, day, time, or specialization.

- **Read Appointment**:
    - This function allows users to view their appointments.
    - The system will return appointments details, including doctor name, day, time, and specialization.

### **Tools for disease and condition Information**:
- **TavilySearchResults**: Search for health, diet, and nutrition information using the `TAVILY_API_KEY` for API calls.
- **WebBaseLoader**:
  - `loader1`: Extract data for disease and patient conditions(https://www.mayoclinic.org/diseases-conditions).
  - Combine content into `docs1` for a comprehensive perspective.
  - Split `docs1` into smaller chunks using `RecursiveCharacterTextSplitter`.
  - Use `FAISS` to create a retriever tool for querying relevant content.

### **Automatic Email Sending on Appointment Booking**:
- After successfully booking an appointment:
  - Compose an email containing:
    - Doctor’s name
    - Specialization
    - Appointment day and time
    - Hospital name (if available)
  - Use email sending functionality to dispatch a confirmation to the user's provided email address.
  - Ensure the email is polite, professional, and confirms all relevant details.


### **Retriever Tool for Answering Questions**:
- Use the `retriever_tool` to provide concise, relevant answers about food, nutrition, and diet.
- Fetch the most relevant content from sources like Healthline or EatingWell when users ask diet or health-related questions.

### **Guidelines for Interaction**:
1. Always maintain a conversational and polite tone.
2. Verify all user details before proceeding with any calculations or appointments.
3. Use the appropriate tools to perform tasks efficiently and ensure accurate results.
4. Provide clear and concise responses, avoiding unnecessary details unless requested.

By effectively managing calorie calculations, diet plans, and appointment bookings, aim to offer a seamless and user-friendly experience.''')


# Node
def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder: StateGraph = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
memory: MemorySaver = MemorySaver()
react_graph_memory: CompiledStateGraph = builder.compile(checkpointer=memory)

# class UserInput(BaseModel):
#     input_text: str 

# # API endpoint
# @app.post("/generateanswer")
# async def generate_answer(user_input: UserInput):
#     try:
#         messages = [HumanMessage(content=user_input.input_text)]
#         response = react_graph_memory.invoke({"messages": messages}, config={"configurable": {"thread_id": "1"}})

#         # Extract the response from the graph output
#         if response and "messages" in response:
#             # Extract the last message (assistant's response)
#             assistant_response = response["messages"][-1].content
#             return {"response": assistant_response}
#         else:
#             return {"response": "No response generated."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the application
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)



# # Specify a thread
# config1 = {"configurable": {"thread_id": "1"}}


# messages = [HumanMessage(content="How much would I save by switching to solar panels if my monthly electricity cost is $200?")]
# messages = react_graph_memory.invoke({"messages": messages}, config1)
# for m in messages['messages']:
#     m.pretty_print()


# Chainlit event handlers
@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    # Set custom theme
    cl.user_session.set(
        "theme",
        {
            "palette": {
                "primary": "#4CAF50",  # Green for health theme
                "background": "#F5F5F5",  # Light background
                "text": "#333333",  # Dark text
                "paper": "#FFFFFF",  # White chat bubbles
            },
            "layout": {
                "fontFamily": "'Roboto', sans-serif",
                "borderRadius": 10,
                "padding": 15,
            }
        }
    )
    await cl.Message(
        content="Welcome to the Telemedicine Asistant BOt! How can I assist you today?",
        author="Assistant"
    ).send()
    cl.user_session.set("chat_history", [])
    cl.user_session.set("thread_id", "1")

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    msg = cl.Message(content="Thinking...", author="Assistant")
    await msg.send()

    # Retrieve and update chat history
    history = cl.user_session.get("chat_history") or []
    history.append(HumanMessage(content=message.content))

    try:
        # Invoke LangGraph with the current history
        config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
        response = react_graph_memory.invoke({"messages": history}, config)

        # Extract the assistant's response
        if response and "messages" in response:
            assistant_response = response["messages"][-1].content
            # Format response with markdown for better readability
            formatted_response = f"**Response:** {assistant_response}"
            msg.content = formatted_response
            await msg.update()

            # Update chat history
            cl.user_session.set("chat_history", response["messages"])

            # Log interaction
            print(f"User: {message.content}")
            print(f"Assistant: {assistant_response}")
        else:
            msg.content = "No response generated."
            await msg.update()

    except Exception as e:
        msg.content = f"**Error:** {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")