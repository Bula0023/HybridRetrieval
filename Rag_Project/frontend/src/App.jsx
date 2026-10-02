import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'
function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading]= useState(false);
  const [error, setError] = useState("");
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("")
  const [answer, setAnswer] = useState("");

  async function uploadFile(){
    if (!file) return;
    const formData = new FormData();

    formData.append("file", file);
    console.log("File being uploaded:", file);
    const response = await fetch(
      "http://127.0.0.1:8000/documents",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await response.json()
    console.log("Upload response:", data);

    if (!response.ok) {
      setUploadStatus("Upload failed");
      return;
    }
    setUploadStatus(
      `${data.filename} uploaded successfully`
    )

  }

  async function askQuestion(){
      try {
          setLoading(true);
          setError("");

          const response = await fetch(
              `http://127.0.0.1:8000/query?query=${encodeURIComponent(question)}`,
              {
                  method: "POST",

              }
          );
          if (!response.ok){
              throw new Error("Query failed");
          }

          // const data = await response.json();
          //   // See entire FastAPI response
          // console.log("Query response:", data);
          // setResult(data);
          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, {
              stream: true
            });

            const lines = buffer.split("\n");

            buffer = lines.pop();

            for (const line of lines) {
              if (!line.trim()) continue;

              const event = JSON.parse(line);

              if (event.type === "answer") {
                setAnswer(previous => previous + event.content);
              }

              if (event.type === "confidence") {
                setResult(event.data);
              }
            }
          }
      } catch (err) {
          setError(err.message);
      } finally {
          setLoading(false);
      }
  }
  return (
      <div>
    <h1>RAG Query Dashboard</h1>

    <section>
      <input 
        type="file"
        accept=".pdf,.text,.md,.html,.htm"
        onChange={(e) =>
          setFile(e.target.files[0])
        }
        />
      <button onClick={uploadFile}>
        Upload
      </button>
      <p>{uploadStatus}</p>
    </section>
    <section>
    <input
      value={question}
      onChange={(e) => setQuestion(e.target.value)}
      placeholder="Ask a question..."
    />

    <button onClick={askQuestion}>
      Ask
    </button>

    {loading && <p>Searching...</p>}

    {error && <p>{error}</p>}

    {answer && (
        <div>
          <h2>Answer</h2>
          <p>{answer}</p>
        </div>
      )}

      {result && (
        <div>
          <h2>Confidence</h2>
          <p>{result.confidence}</p>
        </div>
      )}
    </section>
  </div>
  
  )
}
export default App;
// function App() {
//   const [count, setCount] = useState(0)

//   return (
//     <>
//       <section id="center">
//         <div className="hero">
//           <img src={heroImg} className="base" width="170" height="179" alt="" />
//           <img src={reactLogo} className="framework" alt="React logo" />
//           <img src={viteLogo} className="vite" alt="Vite logo" />
//         </div>
//         <div>
//           <h1>Get started</h1>
//           <p>
//             Edit <code>src/App.jsx</code> and save to test <code>HMR</code>
//           </p>
//         </div>
//         <button
//           type="button"
//           className="counter"
//           onClick={() => setCount((count) => count + 1)}
//         >
//           Count is {count}
//         </button>
//       </section>

//       <div className="ticks"></div>

//       <section id="next-steps">
//         <div id="docs">
//           <svg className="icon" role="presentation" aria-hidden="true">
//             <use href="/icons.svg#documentation-icon"></use>
//           </svg>
//           <h2>Documentation</h2>
//           <p>Your questions, answered</p>
//           <ul>
//             <li>
//               <a href="https://vite.dev/" target="_blank">
//                 <img className="logo" src={viteLogo} alt="" />
//                 Explore Vite
//               </a>
//             </li>
//             <li>
//               <a href="https://react.dev/" target="_blank">
//                 <img className="button-icon" src={reactLogo} alt="" />
//                 Learn more
//               </a>
//             </li>
//           </ul>
//         </div>
//         <div id="social">
//           <svg className="icon" role="presentation" aria-hidden="true">
//             <use href="/icons.svg#social-icon"></use>
//           </svg>
//           <h2>Connect with us</h2>
//           <p>Join the Vite community</p>
//           <ul>
//             <li>
//               <a href="https://github.com/vitejs/vite" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#github-icon"></use>
//                 </svg>
//                 GitHub
//               </a>
//             </li>
//             <li>
//               <a href="https://chat.vite.dev/" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#discord-icon"></use>
//                 </svg>
//                 Discord
//               </a>
//             </li>
//             <li>
//               <a href="https://x.com/vite_js" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#x-icon"></use>
//                 </svg>
//                 X.com
//               </a>
//             </li>
//             <li>
//               <a href="https://bsky.app/profile/vite.dev" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#bluesky-icon"></use>
//                 </svg>
//                 Bluesky
//               </a>
//             </li>
//           </ul>
//         </div>
//       </section>

//       <div className="ticks"></div>
//       <section id="spacer"></section>
//     </>
//   )
// }

// export default App
