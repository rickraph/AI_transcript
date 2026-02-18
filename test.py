import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: API Key not found in .env")
else:
    print(f"✅ API Key found: {api_key[:5]}*******")
    
    try:
        client = genai.Client(api_key=api_key)
        print("\n🔍 Fetching available models for you...")
        
        # List all models
        for m in client.models.list():
            if "generateContent" in m.supported_actions:
                print(f"   - {m.name}")
                
    except Exception as e:
        print(f"\n❌ connection Error: {e}")