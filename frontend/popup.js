const API = "http://127.0.0.1:8000";

const statusEl = document.getElementById("status");
const chooseBtn = document.getElementById("chooseBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const fileInput = document.getElementById("videoInput");
const fileNameEl = document.getElementById("fileName");

chooseBtn.onclick = () => {
  fileInput.click();
};

fileInput.onchange = () => {
  const file = fileInput.files?.[0];
  fileNameEl.innerText = file ? file.name : "No file chosen";
};

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

    const res = await fetch(API + "/analyse", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Upload failed: ${res.status}`);
    }

    const data = await res.json();
    if (typeof data.final_score !== "undefined") {
      statusEl.innerText = `Final: ${data.verdict} (${data.final_score})`;
    } else {
      statusEl.innerText = "Analysis completed";
    }
  } catch (err) {
    statusEl.innerText = "Backend not reachable";
    console.error(err);
  }
};
