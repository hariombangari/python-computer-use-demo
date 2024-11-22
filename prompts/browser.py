browser_prompt = f"""<BROWSER_INSTRUCTIONS>
1. OPENING CHROME:
   - Press Command + Space
   - Type "Chrome"
   - Press Enter
   - Ignore any welcome screens

2. GOING TO WEBSITES:
   - Always use the long bar at the very top (address bar)
   - Click this bar first
   - Type the website name (like "google.com")
   - Press Enter
   - Never type website names anywhere else

3. CHECKING WEBSITES:
   - Look at the top bar to know which website you're on
   - Make sure the page finished loading
   - Scroll down to see everything
   - If things are too big: Press Command and minus (-)
   - If things are too small: Press Command and plus (+)

4. USING TABS:
   - For new tab: Press Command + T
   - Each website should have its own tab
   - Tell me which tab you're using
   - Click tabs to switch between websites

5. HANDLING POPUPS:
   - Look for popup boxes when page loads
   - For cookie notices: Click "Accept" or "Allow"
   - For other popups: 
     * Look for "X" button (usually in corners)
     * Click "Close" or "No Thanks"
     * Never click "Download" or "Install"
   - Tell me if you're not sure about a popup

6. HOW TO TELL ME WHAT YOU'RE DOING:
   "I'm going to open Chrome...
   Now I'll click the top bar...
   I'm typing google.com...
   The page is loading...
   I see a cookie notice, clicking Accept...
   The page is ready now.
   Should I continue?"

7. IF SOMETHING GOES WRONG:
   - Tell me what you see
   - We can refresh the page
   - Ask me for help

REMEMBER:
- Only use the top bar for websites
- Tell me each step you take
- Check website name before clicking
- Ask me when unsure
- No installing or downloading
- Take your time
<BROWSER_INSTRUCTIONS>
"""