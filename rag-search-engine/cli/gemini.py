import os
from dotenv import load_dotenv
from google import genai

def load_gemini():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") 
    client = genai.Client(api_key=api_key)
    return client

def spell_check(text):
    client = load_gemini()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{text}"

If no errors, return the original query.
Corrected:""")
    return response.text