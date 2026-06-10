const BACKEND_URL = "http://localhost:8000";

document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const query = input.value;
    if (!query) return;

    appendMessage("user", query);
    input.value = "";

    const messageDiv = appendMessage("assistant", "Thinking...");

    const response = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: query,
            chunk_size: parseInt(document.getElementById("chunk_size").value),
            chunk_overlap: parseInt(document.getElementById("chunk_overlap").value)
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantText = "";
    messageDiv.innerHTML = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        for (const line of lines) {
            if (line.startsWith("__METADATA__:")) {
                const metadata = JSON.parse(line.replace("__METADATA__:", ""));
                renderSources(messageDiv, metadata);
            } else if (line) {
                assistantText += line;
                const textNode = document.createElement("span");
                textNode.innerText = line;
                messageDiv.insertBefore(textNode, messageDiv.querySelector(".sources"));
            }
        }
    }
});

document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const status = document.getElementById("upload-status");
    status.innerText = "Uploading & Indexing...";

    const formData = new FormData();
    formData.append("file", document.getElementById("pdf-file").files[0]);
    formData.append("chunk_size", document.getElementById("chunk_size").value);
    formData.append("chunk_overlap", document.getElementById("chunk_overlap").value);

    try {
        const res = await fetch(`${BACKEND_URL}/index-pdf`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        status.innerText = data.message || "Success!";
    } catch (err) {
        status.innerText = "Error indexing PDF.";
    }
});

document.getElementById("run-eval").addEventListener("click", async () => {
    const status = document.getElementById("eval-status");
    status.innerText = "Running evaluation suite (please wait)...";

    const formData = new FormData();
    formData.append("chunk_size", document.getElementById("chunk_size").value);
    formData.append("chunk_overlap", document.getElementById("chunk_overlap").value);

    try {
        const res = await fetch(`${BACKEND_URL}/evaluate`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        status.innerText = "Evaluation Completed!";

        document.getElementById("results-card").classList.remove("hidden");
        document.getElementById("val-faithfulness").innerText = data.metrics.faithfulness.toFixed(2);
        document.getElementById("val-relevance").innerText = data.metrics.relevance.toFixed(2);
        document.getElementById("val-retrieval").innerText = (data.metrics.retrieval_accuracy * 100).toFixed(0) + "%";

        renderEvalTable(data.details);
    } catch (err) {
        status.innerText = "Error running evaluation.";
    }
});

function appendMessage(sender, text) {
    const chatMessages = document.getElementById("chat-messages");
    const div = document.createElement("div");
    div.className = `chat-message ${sender}`;
    div.innerHTML = `<strong>${sender === 'user' ? 'You' : 'System'}:</strong> <span>${text}</span>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function renderSources(messageDiv, metadata) {
    const sourcesDiv = document.createElement("div");
    sourcesDiv.className = "sources";
    sourcesDiv.innerHTML = "<strong>Sources:</strong><br>";
    metadata.forEach((src, idx) => {
        const docName = src.metadata?.source || "Indexed Doc";
        sourcesDiv.innerHTML += `[${idx + 1}] ${docName}: "${src.content.substring(0, 100)}..."<br>`;
    });
    messageDiv.appendChild(sourcesDiv);
}

function renderEvalTable(details) {
    const container = document.getElementById("eval-table-container");
    let html = `<table>
        <thead>
            <tr>
                <th>Question</th>
                <th>Faithfulness</th>
                <th>Relevance</th>
                <th>Retrieval Acc</th>
            </tr>
        </thead>
        <tbody>`;
    details.forEach(item => {
        html += `<tr>
            <td>${item.question}</td>
            <td>${item.faithfulness}</td>
            <td>${item.relevance}</td>
            <td>${(item.retrieval_accuracy * 100).toFixed(0)}%</td>
        </tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;
}
