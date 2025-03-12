import React, { userState, useState } from "react";

function App() {
  const [inputText, setInputText] = useState("");
  const [responseText, setResponseText] = useState("");
  const [loading, setLoading] = useState(false);


  const handleSubmit = async (e) => {
    e.preventDefault();

    const userMessage = inputText;
    const requestBody = { text: userMessage};

    try{
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error("HTTP error! Status: ${response.status}");
      }
      const responseData = await response.json();
      setResponseText(responseData.generated_text);
    } catch (error) {
      console.error("Error feching AI response:", error);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "auto" }}>
      <h2>ChatGPT-like AI</h2>
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        rows="4"
        cols="50"
        placeholder="入力してください"
        />
        <br />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "生成中...":"送信"}
        </button>
        <div>
          <h3>AIの回答：</h3>
          <p>{responseText}</p>
        </div>
    </div>
  );
}

export default App;
