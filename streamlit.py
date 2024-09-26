from main import ChatBot
import streamlit as st
import re

bot = ChatBot()

def extract_answer(response):
        
        answer = re.search(r"Answer:\s*(.*)", response, re.DOTALL)
        if answer:
            
            return answer.group(1).strip()
        else:
            return "Sorry, I couldn't extract the answer."

st.set_page_config(page_title="CPE-UNIBEN Chat Bot")
with st.sidebar:
    st.title('Ask Me About Computer Engineering')
    st.title('This project was completed by the Members of Group 9 in CPE461 22/23 session ')


def generate_response(input):
    try:
        
        if not input:
            raise ValueError("Input cannot be empty or None.")
        
        result = bot.rag_chain.invoke(input)
        

        if result is None:
            raise ValueError("The bot returned no response.")
        

        return result.strip()      
    except ValueError as ve:
   
        if "token seems invalid" in str(ve):
            st.error(f"Authorization failed: {str(ve)}")
            return "It seems the authorization token is invalid. Please refresh or log in again."
        else:
            st.error(f"ValueError: {str(ve)}")
            return "There was an issue with your input or the response."
    
    except AttributeError as e:
        st.error(f"An AttributeError occurred: {str(e)}")
        return "Sorry, something went wrong while processing your request."
    
    except Exception as ex:
        st.error(f"An unexpected error occurred: {str(ex)}")
        return "Oops! Something went wrong. Please try again later."


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "CPE Bot", "content": "Welcome to The Department of Computer Engineering Online Assistance Artificial Intelligence Bot, let's begin by you asking me some questions about the department"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

input = st.chat_input()  

# Proceed only if input is valid (i.e., not None or empty)
if input:
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)
        
        
        
    

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("CPE Bot"):
            with st.spinner("Getting your answer from the Department of Computer Engineering Database..."):
                response = generate_response(input)
                
                # Extract only the answer part using the regular expression
                answer = extract_answer(response)
                
                # Display only the answer
                st.write(answer)
        
        message = {"role": "CPE BOT", "content": answer}
        st.session_state.messages.append(message)

    
    