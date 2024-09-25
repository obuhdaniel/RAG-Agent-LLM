from main import ChatBot
import streamlit as st
import re

bot = ChatBot()

def extract_answer(response):
        # Regular expression to match the answer after the 'Answer:' keyword
        answer = re.search(r"Answer:\s*(.*)", response, re.DOTALL)
        if answer:
            # Return the part after 'Answer:'
            return answer.group(1).strip()
        else:
            return "Sorry, I couldn't extract the answer."

st.set_page_config(page_title="CPE-UNIBEN Chat Bot")
with st.sidebar:
    st.title('Ask Me About Computer Engineering')
    st.title('This project was completed by the Memebers of Group 9 in CPE461 22/23 session ')


def generate_response(input):
    try:
        # Ensure input is a valid string and not None
        if not input:
            raise ValueError("Input cannot be empty or None.")
        
        # Invoke the bot's response
        result = bot.rag_chain.invoke(input)
        
        # Check if the result is None
        if result is None:
            raise ValueError("The bot returned no response.")
        
        # Assuming result is a string containing the entire output (including context, question, and answer)
        # We extract only the answer part
        return result.strip()  # Assuming the model returns just the final answer; adjust if necessary.
    
    except ValueError as ve:
        # Handling invalid token or authorization issues
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

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "CPE Bot", "content": "Welcome to The Department of Computer Engineering Online Assistance Artificial Intelligence Bot, let's begin by you asking me some questions about the department"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
input = st.chat_input()  # Fetch input from the user

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

    
    