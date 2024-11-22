google_prompt = f"""<GOOGLE_SEARCH_INSTRUCTIONS>
BEFORE ANY SEARCH:
1. Check Where You Are:
   - Look at the address bar at the top
   - It MUST say "google.com" or "www.google.com"
   - If you're on any other website, DO NOT try to search there

2. If You're Not on Google.com:
   - Open a new tab (Command + T)
   - Type "google.com" in the address bar
   - Press Enter
   - Wait for Google to load
   - Look for the Google logo and search box

SAFE SEARCH RULES:
1. ONLY search if you see:
   - The Google logo
   - The big search box in the middle
   - "google.com" in the address bar

2. NEVER search if:
   - You're on a different website
   - You don't see the Google logo
   - The address bar shows any other website
   - You're not sure if you're on Google

HOW TO TELL ME WHAT YOU'RE DOING:
"Let me check where I am...
I see I'm on [website name].
I need to go to Google first.
Opening new tab with Command + T.
Typing 'google.com'...
...
Now I'm on Google.com - I can see the logo and search box.
Should I do the search now?"

REMEMBER:
- Always check the website first
- Only search on Google.com
- When in doubt, open a new tab
- Tell me what website you're on
- Ask before searching

IF YOU'RE NOT ON GOOGLE:
"I notice I'm not on Google.com right now.
I'm on [other website].
I'll open a new tab to go to Google first."

This way, we stay safe and only search on the real Google website!
<GOOGLE_SEARCH_INSTRUCTIONS>
"""