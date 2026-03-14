const API_URL = "http://127.0.0.1:8000/infer";

const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const preview = document.getElementById("preview");
const galleryInput = document.getElementById("galleryInput");

let imageBlob = null;

// Open Camera
function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(() => alert("Camera access denied"));
}

// Capture
function capturePhoto() {
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        imageBlob = blob;
        preview.innerHTML = `<img src="${URL.createObjectURL(blob)}">`;
    }, "image/png");
}

// Upload
galleryInput.addEventListener("change", () => {
    imageBlob = galleryInput.files[0];
    preview.innerHTML = `<img src="${URL.createObjectURL(imageBlob)}">`;
});

// Send to Backend
async function sendImage() {

    if (!imageBlob) {
        alert("Upload or capture image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", imageBlob);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        displayResult(data);

    } catch (error) {
        alert("Backend not running!");
        console.error(error);
    }
}

// Display Result
function displayResult(data) {

    const stateEl = document.getElementById("state");
    const confidenceEl = document.getElementById("confidence");
    const originalDaysEl = document.getElementById("originalDays");
    const tempEl = document.getElementById("temperature");
    const improvedEl = document.getElementById("improvedDays");
    const bar = document.getElementById("shelfBar");

    stateEl.innerText = data.state;
    confidenceEl.innerText = data.confidence;
    originalDaysEl.innerText = data.original_days_left;
    tempEl.innerText = data.recommended_temperature;
    improvedEl.innerText = data.improved_shelf_life_days;

    if (data.state.includes("fresh")) {
        stateEl.className = "fresh";
        bar.style.background = "#2e7d32";
    } else if (data.state.includes("rotten")) {
        stateEl.className = "rotten";
        bar.style.background = "#c62828";
    }

    const percent = Math.min((data.original_days_left / 10) * 100, 100);
    bar.style.width = percent + "%";

    document.getElementById("result").classList.remove("hidden");
}