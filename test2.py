import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def test_connection():
    # 1. Verify the environment variable exists
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY environment variable is missing.")
        print("Fix: Run 'export GROQ_API_KEY=your_key_here' in your terminal.")
        return

    client = Groq(api_key=api_key)

    print(f"Attempting to connect with key: {api_key[:6]}...{api_key[-4:]}")

    try:
        # 2. Try a low-latency model for a quick ping
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Connected'",
                }
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=10,
        )
        
        result = chat_completion.choices[0].message.content.strip()
        print(f"✅ SUCCESS: Groq is online. Response: {result}")

    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")

if __name__ == "__main__":
    test_connection()