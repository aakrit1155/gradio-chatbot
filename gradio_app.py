import os
import traceback  # To catch potential unexpected errors during validation

import gradio as gr
from anthropic import AuthenticationError as AnthropicAuthenticationError
from dotenv import load_dotenv
from google.api_core.exceptions import ClientError as GoogleClientError
from google.api_core.exceptions import PermissionDenied as GooglePermissionDenied
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Import potential exception types for validation
from openai import AuthenticationError as OpenAIAuthenticationError

# --- Configuration ---
load_dotenv()  # Load default keys if available, though user input is primary

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# load css from style.css
with open(os.path.join(CURRENT_DIR, "style.css"), "r", encoding="utf-8") as f:
    css = f.read()


initial_dropdown_prompt = "-- Select LLM --"
choices = [
    initial_dropdown_prompt,
    "OPENAI ChatGPT",
    "Google GenAI",
    "Anthropic Claude",
]

llm_env_vars = {
    "OPENAI ChatGPT": "OPENAI_API_KEY",
    "Google GenAI": "GOOGLE_API_KEY",
    "Anthropic Claude": "ANTHROPIC_API_KEY",
}
llm_models = {
    "OPENAI ChatGPT": "gpt-4o-mini",
    "Google GenAI": "gemini-1.5-flash",
    "Anthropic Claude": "claude-3-haiku-20240307",
}

system_message_content = """You are an intelligent chatbot who talks like a pirate in an archaic English manner. Be jovial and use plenty of pirate slang like 'Ahoy!', 'Matey', 'Shiver me timbers!', 'Landlubber', etc."""

# --- LLM Initialization Functions (Modified for Key Input & Validation) ---


def initialize_llm(llm_choice: str, api_key: str):
    """
    Attempts to initialize the selected LLM with the provided API key.
    Returns the initialized client on success, None on failure.
    Also returns an error message string if initialization fails.
    """
    try:
        if llm_choice == "OPENAI ChatGPT":
            print(f"Attempting to initialize OpenAI with key: ...{api_key[-4:]}")
            client = ChatOpenAI(
                temperature=0.7,
                model=llm_models[llm_choice],
                streaming=True,
                api_key=api_key,
            )
            # Add a simple test call to ensure the key is valid
            client.invoke("Ahoy!")
            print("OpenAI client initialized successfully.")
            return client, None

        elif llm_choice == "Google GenAI":
            print(f"Attempting to initialize Google GenAI with key: ...{api_key[-4:]}")
            client = ChatGoogleGenerativeAI(
                model=llm_models[llm_choice],
                google_api_key=api_key,
                temperature=0.7,  # Add temperature if desired
                # convert_system_message_to_human=True # Might be needed depending on model/Langchain version
            )
            # Add a simple test call
            client.invoke("Ahoy!")
            print("Google GenAI client initialized successfully.")
            return client, None

        elif llm_choice == "Anthropic Claude":
            print(f"Attempting to initialize Anthropic with key: ...{api_key[-4:]}")
            client = ChatAnthropic(
                model=llm_models[llm_choice],
                anthropic_api_key=api_key,
                temperature=0.7,  # Add temperature if desired
            )
            # Add a simple test call
            client.invoke("Ahoy!")
            print("Anthropic client initialized successfully.")
            return client, None

        else:
            return None, "Invalid LLM choice selected."

    # Specific Authentication Errors
    except OpenAIAuthenticationError:
        print("OpenAI Authentication Error")
        return None, "Invalid OpenAI API Key. Shiver me timbers! Check yer key."
    except (GooglePermissionDenied, GoogleClientError) as e:
        # Google might raise PermissionDenied or other ClientErrors for bad keys/setup
        print(f"Google API Error: {e}")
        return (
            None,
            f"Invalid Google API Key or configuration error. Blimey! ({type(e).__name__})",
        )
    except AnthropicAuthenticationError:
        print("Anthropic Authentication Error")
        return None, "Invalid Anthropic API Key. Arrgh! That key be wrong."
    # Catch other potential errors during initialization/test call
    except Exception as e:
        print(f"An unexpected error occurred during LLM initialization: {e}")
        print(traceback.format_exc())  # Print full traceback for debugging
        error_type = type(e).__name__
        return (
            None,
            f"An unexpected error occurred ({error_type}). Check console logs. Walk the plank!",
        )


# --- Gradio UI Functions ---


def update_visibility_after_llm_select(llm_choice):
    """Shows the API key input section when an LLM is selected."""
    if llm_choice and llm_choice != initial_dropdown_prompt:
        key_placeholder = f"Enter yer {llm_choice} API Key, ye scurvy dog!"
        # Show API key row, hide chat row
        return {
            api_key_row: gr.update(visible=True),
            api_key_textbox: gr.update(
                placeholder=key_placeholder, value=""
            ),  # Clear previous key
            validation_status_msg: gr.update(value="", visible=False),  # Hide status
            chat_interface_row: gr.update(visible=False),
        }
    else:
        # Hide both if dropdown is cleared
        return {
            api_key_row: gr.update(visible=False),
            validation_status_msg: gr.update(value="", visible=False),
            chat_interface_row: gr.update(visible=False),
        }


def validate_api_key_and_setup_chat(llm_choice, api_key, current_state):
    """
    Validates the API key. If valid, hides setup UI and shows chat UI.
    Updates the state with the validated client.
    """
    if not api_key:
        return {
            validation_status_msg: gr.update(
                value="Avast! Ye need to enter an API key!", visible=True
            ),
            llm_client_state: current_state,  # Keep state as is
        }

    # Update status to show validation is in progress
    yield {
        validation_status_msg: gr.update(
            value="Aye, checkin' yer key...", visible=True, elem_classes=["success"]
        ),
        validate_button: gr.update(
            interactive=False
        ),  # Disable button during validation
    }

    llm_client, error_message = initialize_llm(llm_choice, api_key)

    if llm_client and error_message is None:
        print(f"Validation successful for {llm_choice}")
        # Key is valid: Hide setup, show chat
        updates = {
            llm_select_row: gr.update(visible=False),
            api_key_row: gr.update(visible=False),
            validation_status_msg: gr.update(
                value="Key be valid! Let's chat!", visible=True, elem_classes=["error"]
            ),  # Optional success msg
            chat_interface_row: gr.update(visible=True),
            chatbot: gr.update(value=[]),
            # chat_history_state: [],  # Reset chat history for the new session
            llm_client_state: llm_client,  # Store the validated client in state!
        }
    else:
        print(f"Validation failed for {llm_choice}: {error_message}")
        # Key is invalid: Show error, keep setup visible, clear client state
        updates = {
            validation_status_msg: gr.update(
                value=f"Validation Failed: {error_message}", visible=True
            ),
            api_key_textbox: gr.update(value=""),  # Clear the invalid key
            validate_button: gr.update(interactive=True),  # Re-enable button
            llm_client_state: None,  # Ensure client state is None
        }

    yield updates


def llm_response(message, history, llm_client):
    """Generates and streams the chat response using the validated LLM client."""
    if llm_client is None:
        yield [
            {
                "role": "assistant",
                "content": "Shiver me timbers! The LLM client be not initialized. Refresh the page and try again.",
            }
        ]
        return

    print(f"Chat Input:-> {message}")
    print(f"History (type='messages' format): {history}")

    # Prepend the system message ONCE per session effectively
    # Though we format it every time, it only matters semantically for the LLM
    history_langchain_format = [SystemMessage(content=system_message_content)]

    if isinstance(history, list):
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role == "user":
                history_langchain_format.append(HumanMessage(content=content))
            elif role == "assistant":
                history_langchain_format.append(AIMessage(content=content))
            # Handle potential unexpected format, though unlikely with gr.Chatbot
            elif (
                content
            ):  # If role is missing but content exists, treat as human? Or skip?
                print(f"Warning: History item missing role: {item}. Skipping.")

    # Add the current user message
    history_langchain_format.append(HumanMessage(content=message))

    history.append({"role": "user", "content": str(message)})
    history.append({"role": "assistant", "content": ""})
    # Stream the response
    full_response = ""
    try:
        # Use the llm_client passed via state
        for chunk in llm_client.stream(history_langchain_format):
            if chunk.content:
                full_response += chunk.content

        history[-1][
            "content"
        ] = full_response  # Update the last message with the full response
        print("The pirate response: ", full_response)
        yield history

    except Exception as e:
        print(f"Error during LLM stream: {e}")
        print(traceback.format_exc())
        error_chat_msg = {
            "role": "assistant",
            "content": f"Blast it all! An error occurred while talkin' to the LLM: {type(e).__name__}",
        }
        history.append(error_chat_msg)
        yield history


# --- Gradio Application Build ---

with gr.Blocks(
    theme=gr.themes.Glass(), title="Pirate LLM Chat", css=css, elem_id="main-container"
) as demo:
    # State variables - crucial for session management
    # llm_client_state holds the validated Langchain client object
    llm_client_state = gr.State(None)
    # chat_history_state holds the conversation for gr.Chatbot
    chat_history_state = gr.State([])

    gr.Markdown(
        "# Ahoy! Chat with a Pirate LLM\nSelect yer desired LLM crewmate below.",
        elem_id="title-markdown",
    )

    # Stage 1: LLM Selection
    with gr.Row(
        visible=True, elem_id="llm-select-section", elem_classes=["content-row"]
    ) as llm_select_row:
        llm_dropdown = gr.Dropdown(
            choices=choices,
            value=initial_dropdown_prompt,
            label="Choose yer LLM",
            info="Which digital scallywag ye want to talk to?",
            elem_classes=["input-field", "llm-selector"],
            interactive=True,
        )

    # Stage 2: API Key Input & Validation
    with gr.Row(
        visible=False, elem_id="api-key-section", elem_classes=["content-row"]
    ) as api_key_row:  # Initially hidden
        with gr.Column(scale=4, elem_classes=["api-key-column"]):
            api_key_textbox = gr.Textbox(
                label="API Key",
                placeholder="Enter yer API Key here, matey!",
                type="password",
                interactive=True,
                elem_classes=["input-field", "api-key-input"],
            )
        with gr.Column(scale=1, elem_classes=["validate-button-column"], min_width=120):
            validate_button = gr.Button(
                "Validate Key", elem_classes=["action-button", "validate-button"]
            )

    # Status Message Area (for validation feedback)
    validation_status_msg = gr.Markdown(
        value="", visible=False, elem_id="validation-status"
    )

    # Stage 3: Chat Interface
    with gr.Row(
        visible=False, elem_id="chat-section"
    ) as chat_interface_row:  # Initially hidden
        with gr.Column():
            chatbot = gr.Chatbot(
                label="Pirate Chat",
                show_copy_all_button=True,
                autoscroll=True,
                height=500,
                type="messages",
                container=True,
                elem_id="chatbot-display",
                elem_classes=["chat-display"],
                # value=chat_history_state # Bind directly if possible, or manage via function outputs
            )
            chat_input = gr.Textbox(
                placeholder="Type yer message to the pirate bot here...",
                container=False,
                scale=7,
                elem_classes=["input-field", "chat-message-input"],
            )
            # submit_button = gr.Button("Send", scale=1) # ChatInterface includes submit

    # --- Event Handling Logic ---

    # 1. When LLM dropdown changes -> show API key input
    llm_dropdown.change(
        fn=update_visibility_after_llm_select,
        inputs=[llm_dropdown],
        outputs=[
            api_key_row,
            api_key_textbox,
            validation_status_msg,
            chat_interface_row,
        ],
    )

    # 2. When Validate button is clicked -> attempt validation
    validate_button.click(
        fn=validate_api_key_and_setup_chat,
        inputs=[llm_dropdown, api_key_textbox, llm_client_state],  # Pass current state
        outputs=[  # Update multiple components based on validation result
            llm_select_row,
            api_key_row,
            validation_status_msg,
            chat_interface_row,
            api_key_textbox,  # To clear on failure
            validate_button,  # To disable/enable
            llm_client_state,  # Update the client state
            chatbot,  # Clear chatbot display on success
        ],
    )
    # chat_history_state,  # Reset history on success

    # 3. When user submits message in chat input
    chat_input.submit(
        fn=llm_response,
        inputs=[
            chat_input,
            chatbot,
            llm_client_state,
        ],  # Use chatbot for history, pass client state
        outputs=[chatbot],  # Stream output to chatbot
    ).then(
        lambda: gr.update(value=""),  # Clear input textbox after submit
        inputs=None,
        outputs=[chat_input],
    )

    # Add clear button functionality if desired (using chatbot directly)
    # clear_button = gr.Button("Clear Chat History")
    # clear_button.click(lambda: [], None, chatbot) # Clears the chatbot display


if __name__ == "__main__":
    demo.launch(debug=True)  # Debug=True helps see logs and errors
