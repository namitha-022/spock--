const API = "http://127.0.0.1:8000";

const statusEl = document.getElementById("status");
const fileInput = document.getElementById("videoInput");
const analyzeBtn = document.getElementById("analyzeBtn");

function connectWs(result) {
  const ws = new WebSocket(`ws://127.0.0.1:8000/ws/${result.analysis_id}`);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      video_task_id: result.video_task_id,
      audio_task_id: result.audio_task_id,
      metadata_task_id: result.metadata_task_id
    }));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.stage === "final") {
        const finalScore = data.result?.final_score;
        statusEl.innerText = finalScore != null
          ? `Final score: ${finalScore}`
          : "Analysis complete";
        return;
      }
      statusEl.innerText = `Stage complete: ${data.stage}`;
    } catch (err) {
      console.error("Invalid WS payload", err);
      statusEl.innerText = "Received invalid update from backend";
    }
  };

  ws.onerror = () => {
    statusEl.innerText = "WebSocket connection error";
  };
}

analyzeBtn.onclick = async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    statusEl.innerText = "Please choose a file first";
    return;
  }

  statusEl.innerText = "Starting analysis...";

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(API + "/analyze-upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error("Request failed with status " + res.status);
    }

    const data = await res.json();
    statusEl.innerText = "Upload accepted. Waiting for results...";
    connectWs(data);
  } catch (err) {
    statusEl.innerText = "Backend not reachable";
    console.error(err);
  }
};
