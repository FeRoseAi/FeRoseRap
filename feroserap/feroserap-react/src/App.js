import React, { userState, useState } from "react";
import './App.css';

function App() {
  const LOADING_TEXT = "loading..."

  const [inputText, setInputText] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  // AIの処理
  async function getAIresponse(userMessage) {
    const requestBody = { text: userMessage};
    try {
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
      return responseData.generated_text;
    } catch (error){
      console.error("Error feching AI response:", error);
    }
  };

  // 送信ボタンが押されたときの処理
  const handleSubmit = async (e) => {
    // 入力が空白なら即処理終了
    if (inputText.trim() === '') return;

    e.preventDefault();

    const userMessage = { sender: "user", message: inputText };
    const loadingMessage = { sender: "ai", message: LOADING_TEXT };

    setChatHistory((prev) => [...prev, userMessage, loadingMessage]);
    setInputText("");
    setLoading(true);

    const aiMessage = await getAIresponse(inputText);

    // AIの返答の更新
    setChatHistory((prev) => {
      const newHistory = [...prev];
      newHistory[newHistory.length - 1] = { sender: "ai", message: aiMessage};
      return newHistory;
    })

    setLoading(false);
  };

  // loadingにアニメーションを追加
  const LoadingMessage = () => (
    <div className="loading">
      {"loading...".split("").map((char, i) => (
        <span key={i} style={{ animationDelay: `${i * 0.1}s` }}>
          {char}
        </span>
      ))}
    </div>
  );

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "180px auto 20px auto" }}>
      <div className="input-area">
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
      </div>

        {chatHistory.map((chat, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: chat.sender === "user" ? "flex-end" : "flex-start",
              margin: "10px 0",
            }}
            >
              <div style={{ fontWaight:"bold" }}>
                {chat.sender === "user" ? "あなた" : "AI" }
              </div>
              <div
                style={{
                  backgroundColor: chat.sender === "user" ? "#dcf8c6" : "#f0ffff",
                  padding: "8px 12px",
                  borderRadius: "10px",
                  maxWidth: "70%",
                  textAlign: "left",
                  whiteSpace: "pre-wrap",
                }}
              >
                {chat.message === LOADING_TEXT ? <LoadingMessage /> : chat.message}
              </div>
          </div>
        ))}
    </div>
  );
}

export default App;
