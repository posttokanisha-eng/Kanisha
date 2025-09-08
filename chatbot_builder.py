import streamlit as st
import os
import re
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_bot" not in st.session_state:
    st.session_state.current_bot = "rule_based"
if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = []
if "bot_personality" not in st.session_state:
    st.session_state.bot_personality = "helpful"
if "selected_template" not in st.session_state:
    st.session_state.selected_template = None
if "bot_config" not in st.session_state:
    st.session_state.bot_config = {}

# Pre-built chatbot templates
CHATBOT_TEMPLATES = {
    "customer_support": {
        "name": "Customer Support Bot",
        "description": "Handles basic customer inquiries and support requests",
        "rules": {
            r'\b(order|tracking|delivery)\b': [
                "I can help you track your order! Please provide your order number.",
                "Let me check on your delivery status. What's your order ID?"
            ],
            r'\b(return|refund)\b': [
                "I understand you'd like to return an item. Our return policy allows returns within 30 days.",
                "For refunds, please contact our billing department or visit our returns page."
            ],
            r'\b(hours|open|closed)\b': [
                "Our customer service is available 24/7 through this chat!",
                "We're here to help anytime! Our phone support is available 9 AM - 6 PM EST."
            ]
        },
        "personality": "professional"
    },
    "educational_tutor": {
        "name": "Educational Tutor Bot",
        "description": "Helps students learn with explanations and encouragement",
        "rules": {
            r'\b(math|algebra|calculus)\b': [
                "Math is fun! What specific topic would you like help with?",
                "I love helping with math problems! Show me what you're working on."
            ],
            r'\b(science|physics|chemistry)\b': [
                "Science is fascinating! What experiment or concept interests you?",
                "Let's explore science together! What would you like to learn about?"
            ],
            r'\b(homework|assignment)\b': [
                "I'm here to help guide you through your homework, not do it for you! What do you need help understanding?",
                "Let's work through this step by step. What part of the assignment is challenging?"
            ]
        },
        "personality": "educational"
    },
    "restaurant_bot": {
        "name": "Restaurant Assistant",
        "description": "Takes orders and provides menu information",
        "rules": {
            r'\b(menu|food|eat|order)\b': [
                "Our menu includes pizza, burgers, salads, and daily specials!",
                "What would you like to order today? We have fresh ingredients!"
            ],
            r'\b(hours|open|closed)\b': [
                "We're open Monday-Sunday, 11 AM - 10 PM!",
                "Our kitchen closes at 9:30 PM, but you can still order until then!"
            ],
            r'\b(reservation|table|book)\b': [
                "I'd be happy to help with reservations! How many people and what time?",
                "Let me check our availability. What day were you thinking?"
            ]
        },
        "personality": "helpful"
    }
}


class RuleBasedChatbot:
    """Simple rule-based chatbot using pattern matching."""

    def __init__(self, template=None):
        # Default rules
        default_rules = {
            r'\b(hello|hi|hey)\b': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! How's it going?"
            ],
            r'\b(bye|goodbye|see you)\b': [
                "Goodbye! Have a great day!",
                "See you later!",
                "Take care!"
            ],
            r'\bname\b': [
                "I'm a rule-based chatbot created to help you learn!",
                "My name is RuleBot. Nice to meet you!"
            ],
            r'\b(thanks|thank you)\b': [
                "You're welcome!",
                "Happy to help!",
                "No problem!"
            ],
            r'\b(help|what can you do)\b': [
                "I can respond to basic greetings, questions about my name, and simple conversations!",
                "I'm a simple rule-based bot. Try saying hello, asking my name, or saying goodbye!"
            ]
        }

        # Load template if provided
        if template and template in CHATBOT_TEMPLATES:
            template_data = CHATBOT_TEMPLATES[template]
            self.rules = {**default_rules, **template_data["rules"]}
            self.name = template_data["name"]
        else:
            self.rules = default_rules
            self.name = "RuleBot"

        self.default_responses = [
            "I'm not sure I understand. Can you rephrase that?",
            "That's interesting! Tell me more.",
            "I'm still learning. Can you ask me something else?",
            "Hmm, I don't have a good response for that yet."
        ]

    def get_response(self, message: str) -> str:
        """Get response based on pattern matching."""
        message_lower = message.lower()

        for pattern, responses in self.rules.items():
            if re.search(pattern, message_lower):
                return random.choice(responses)

        return random.choice(self.default_responses)


class AIChatbot:
    """AI-powered chatbot using OpenAI."""

    def __init__(self, personality: str = "helpful"):
        self.personality_prompts = {
            "helpful": "You are a helpful and friendly assistant. Provide clear, useful responses.",
            "creative": "You are a creative and imaginative assistant. Use vivid language and creative examples.",
            "professional": "You are a professional business assistant. Be formal and direct in your responses.",
            "funny": "You are a witty and humorous assistant. Add appropriate humor to your responses.",
            "educational": "You are an educational tutor. Explain concepts clearly and ask follow-up questions."
        }

        self.personality = personality
        self.client = None

        # Try to initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "sk-default-key":
            self.client = OpenAI(api_key=api_key)

    def get_response(self, message: str, context: List[Dict] = None) -> str:
        """Get AI-powered response."""
        if not self.client:
            return "🔑 Please add your OpenAI API key in the sidebar to use the AI chatbot!"

        try:
            # Prepare system prompt
            system_prompt = self.personality_prompts.get(
                self.personality,
                self.personality_prompts["helpful"]
            )

            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation context
            if context:
                messages.extend(context[-10:])  # Keep last 10 messages for context

            messages.append({"role": "user", "content": message})

            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please check your API key."


def main():
    # Header
    st.title("🤖 AI Chatbot Builder")
    st.markdown("*Learn to build different types of chatbots with hands-on examples*")

    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Chatbot Configuration")

        # Chatbot type selection
        bot_type = st.selectbox(
            "Choose Chatbot Type:",
            ["rule_based", "ai_powered", "custom"],
            index=0,
            format_func=lambda x: {
                "rule_based": "🔧 Rule-Based Bot",
                "ai_powered": "🧠 AI-Powered Bot",
                "custom": "⚡ Custom Bot"
            }[x]
        )

        st.session_state.current_bot = bot_type

        # Configuration based on bot type
        if bot_type == "ai_powered":
            st.subheader("AI Configuration")

            # API Key input
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Enter your OpenAI API key to enable AI responses"
            )

            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("✅ API key configured!")

            # Personality selection
            personality = st.selectbox(
                "Bot Personality:",
                ["helpful", "creative", "professional", "funny", "educational"],
                format_func=lambda x: {
                    "helpful": "😊 Helpful",
                    "creative": "🎨 Creative",
                    "professional": "💼 Professional",
                    "funny": "😄 Funny",
                    "educational": "🎓 Educational"
                }[x]
            )

            st.session_state.bot_personality = personality

        elif bot_type == "rule_based":
            st.subheader("Rule Configuration")

            st.markdown("""
            **Current Rules:**
            - Greetings (hello, hi, hey)
            - Farewells (bye, goodbye) 
            - Name questions
            - Weather questions
            - Thanks/gratitude
            - Help requests
            """)

            # Add custom rule
            st.write("**Add Custom Rule:**")
            pattern = st.text_input("Pattern (regex):", placeholder="e.g., \\bfood\\b")
            response = st.text_input("Response:", placeholder="e.g., I love talking about food!")

            if st.button("Add Rule") and pattern and response:
                st.session_state.custom_rules.append({"pattern": pattern, "response": response})
                st.success("Rule added!")

        elif bot_type == "custom":
            st.subheader("Custom Bot Builder")
            st.markdown("🚧 Coming soon: Visual bot builder with drag-and-drop interface!")

        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")

        # Chat container
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

        # Chat input
        user_input = st.chat_input("Type your message here...")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get bot response based on type
            if st.session_state.current_bot == "rule_based":
                bot = RuleBasedChatbot()

                # Add custom rules
                for custom_rule in st.session_state.custom_rules:
                    bot.rules[custom_rule["pattern"]] = [custom_rule["response"]]

                response = bot.get_response(user_input)

            elif st.session_state.current_bot == "ai_powered":
                bot = AIChatbot(st.session_state.bot_personality)

                # Convert messages to OpenAI format
                context = []
                for msg in st.session_state.messages[-10:]:  # Last 10 messages
                    if msg["role"] in ["user", "assistant"]:
                        context.append(msg)

                response = bot.get_response(user_input, context)

            else:  # custom
                response = "🚧 Custom bot functionality coming soon!"

            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with col2:
        st.header("📚 Learning Center")

        # Tutorial section
        with st.expander("🎓 Chatbot Types", expanded=True):
            st.markdown("""
            **Rule-Based Bots:**
            - Use pattern matching
            - Fast and predictable
            - Limited to predefined responses
            - Great for simple tasks

            **AI-Powered Bots:**
            - Use machine learning models
            - More natural conversations
            - Can handle complex queries
            - Require API keys/costs

            **Hybrid Bots:**
            - Combine both approaches
            - Rule-based for common queries
            - AI for complex requests
            """)

        with st.expander("⚙️ How Rule-Based Bots Work"):
            st.markdown("""
            ```python
            # Pattern matching example
            import re

            def get_response(message):
                if re.search(r'\\bhello\\b', message.lower()):
                    return "Hello! How can I help?"
                elif re.search(r'\\bbye\\b', message.lower()):
                    return "Goodbye!"
                else:
                    return "I don't understand."
            ```
            """)

        with st.expander("🧠 How AI Bots Work"):
            st.markdown("""
            ```python
            # AI chatbot example
            from openai import OpenAI

            client = OpenAI(api_key="your-key")

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": user_message}
                ]
            )
            ```
            """)

        with st.expander("🛠️ Best Practices"):
            st.markdown("""
            **For Rule-Based Bots:**
            - Start with common user intents
            - Use regex for flexible matching
            - Provide fallback responses
            - Test edge cases

            **For AI Bots:**
            - Write clear system prompts
            - Manage conversation context
            - Handle API errors gracefully
            - Set appropriate temperature

            **General Tips:**
            - Keep responses concise
            - Add personality consistently
            - Test with real users
            - Monitor and improve
            """)

        # Quick stats
        st.subheader("📊 Chat Stats")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total", total_messages)
        col_b.metric("User", user_messages)
        col_c.metric("Bot", bot_messages)

    # Footer
    st.markdown("---")
    st.markdown("🤖 **Built with Streamlit** | Learn more about chatbot development!")


if __name__ == "__main__":
    main()