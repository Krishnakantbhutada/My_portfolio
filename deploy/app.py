# imports

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# The usual start

load_dotenv(override=True)
openai = OpenAI()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_pdfsummary(reader):
    txt = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            txt += text

    return txt

reader_1 = PdfReader("Krishnakant_bhutada_Data_Scientist_resume.pdf")
reader_2 = PdfReader("Krishnakant_Linkedin_profile.pdf")

summary = ""

summary += get_pdfsummary(reader_1)
summary += get_pdfsummary(reader_2)

# print(summary)

def record_unknown_question(question):
    print(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

def record_user_details(email, name="Name not provided", notes="not provided"):
    print(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_unknown_question_json},
         {"type": "function", "function": record_user_details_json}]

name = "Krishnakant Bhutada"

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s resume and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

system_prompt += f"\n\n## Summary:\n{summary}"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

# print(system_prompt)


#Tool call 

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
    
        if tool_name == "record_unknown_question":
            result = record_unknown_question(**tool_args)
        elif tool_name == "record_user_details":
            result = record_user_details(**tool_args)

        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    
    return results


# Function to call OpenAI

def chat(message,history):

    done = 1
    messages=[{"role": "system", "content": system_prompt}]+ history + [{"role": "user", "content": message}]

    while(done):
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",   
            messages=messages,
            tools = tools
        )
        
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls
            
            message = response.choices[0].message

            result  = handle_tool_calls(tool_calls)
            print(result)
            messages.append(message)
            messages.extend(result)
        
        else:
            done = 0
        
    

    return response.choices[0].message.content

    


# gr.ChatInterface(chat, type="messages").launch()


import gradio as gr

# Wrapper for normal text input
def handle_user_input(message, history):
    response = chat(message, history)   # call your chat function
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""   # update chat + clear input


# Handle suggestion click
def load_question(question, history):
    response = chat(question, history)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    return history, ""


with gr.Blocks() as demo:
    gr.Markdown("### Chatbot with Suggested Questions")

    chatbot = gr.Chatbot(type="messages", label="Chat Window", height=400)

    # suggestion buttons row
    with gr.Row():
        btn1 = gr.Button("What are your Technical skills")
        btn2 = gr.Button("Give me Summary of your profile")
        btn3 = gr.Button("What are your project highlights")

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Type a message...",
            container=False,
            scale=8,
        )
        send_btn = gr.Button("➤", scale=1)  # small send icon

    # send message either by pressing Enter or clicking the icon
    txt.submit(handle_user_input, [txt, chatbot], [chatbot, txt])
    send_btn.click(handle_user_input, [txt, chatbot], [chatbot, txt])

    # suggestions trigger chat directly
    btn1.click(load_question, [btn1, chatbot], [chatbot, txt])
    btn2.click(load_question, [btn2, chatbot], [chatbot, txt])
    btn3.click(load_question, [btn3, chatbot], [chatbot, txt])

demo.launch()


