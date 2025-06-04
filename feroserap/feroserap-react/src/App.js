import React, { useState, useEffect } from "react";
import './App.css';

function App() {
  const LOADING_TEXT = "loading..."

  const [inputText, setInputText] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [randomSuggestions, setRandomSuggestions] = useState([]);

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
        throw new Error(`HTTP error! Status: ${response.status}`);
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
  )

  // 会話例提示（静的実装）
  // 会話候補
  const suggestions = [
    "今日の天気は？",
    "おすすめの映画を教えて",
    "面白い雑学は？",
    "最近のニュースは？",
    "おすすめの観光地は？"
  ]

  // ランダムに3つ選ぶ
  const getRandomSuggestions = (count = 3) => {
    const shuffled = [...suggestions].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  };

  // 初期表示時にランダム抽出
  useEffect(() => {
    setRandomSuggestions(getRandomSuggestions());
  }, []);

  // 会話例クリック時の処理
  const handleSuggestionClick = async (text) => {
    //setInputText(text);

    const userMessage = { sender: "user", message: text };
    const loadingMessage = { sender: "ai", message: LOADING_TEXT };

    setChatHistory((prev) => [...prev, userMessage, loadingMessage]);
    setLoading(true);

    const aiMessage = await getAIresponse(text);

    // loading を AIの返答に差し替え
    setChatHistory((prev) => {
      const newHistory = [...prev];
      newHistory[newHistory.length - 1] = { sender: "ai", message: aiMessage };
      return newHistory;
    });

    setLoading(false);
  };

  return (
    <div style={{ padding: "0px", maxWidth: "1000px", margin: "0px auto 150px auto" }}>
      <h2>ChatGPT-like AI</h2>
        {chatHistory.length === 0 && (
          <div className="suggestion-area">
            {randomSuggestions.map((text, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(text)}
                className="suggestion-button"
                >
                  {text}
              </button>
            ))}
          </div>
        )}
      <div className="input-area">
        <div className="input-row">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows="3"
            cols="50"
            placeholder="入力してください"
            />
            <button onClick={handleSubmit} disabled={loading}>
              {loading ? "生成中...":"送信"}
            </button>
          </div>
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
