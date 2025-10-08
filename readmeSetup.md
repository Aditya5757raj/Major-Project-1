## 🚀 Project Setup Instructions

1. **Open the terminal** and navigate to your backend folder:

   ```bash
   cd Major-Project-1
   cd backend
   ```

2. **Install all Node.js dependencies:**

   ```bash
   npm i
   ```

3. **Update your MySQL password:**

   * Open the `server.js` file inside the backend folder.
   * Find the MySQL configuration section and update the password field with your own:

     ```js
     password: "your_mysql_password"
     ```

4. **Start the Node.js server:**

   ```bash
   node server.js
   ```

5. **Open a new terminal window** (keep the previous one running) and start the FastAPI server:

   ```bash
   cd backend
   python -m uvicorn server:app --reload --log-level debug
   ```

   If the above doesn’t work, try:

   ```bash
   uvicorn server:app --reload --log-level debug
   ```

6. **Run the frontend:**

   * Open the `index.html` file located in the main folder (`Major-Project-1`) directly in your browser,
     **or** use a local server tool like:

     ```bash
     npx live-server
     ```

7. **That’s it! 🎉**

   * Log in using your credentials
   * Start chatting with the AI
   * All your chats will be saved and displayed automatically in the sidebar
