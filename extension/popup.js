document.addEventListener("DOMContentLoaded", async () => {
  const apiUrlInput = document.getElementById("apiUrl");
  const apiSecretInput = document.getElementById("apiSecret");
  const catModel = document.getElementById("catModel");
  const anModel = document.getElementById("anModel");
  const savedEl = document.getElementById("saved");
  const statusBox = document.getElementById("statusBox");
  const statusMsg = document.getElementById("statusMsg");
  const jobBox = document.getElementById("jobBox");

  const stored = await chrome.storage.sync.get(["apiUrl", "apiSecret"]);
  apiUrlInput.value = stored.apiUrl || "http://localhost:8000";
  apiSecretInput.value = stored.apiSecret || "";

  const local = await chrome.storage.local.get(["lastStatus", "lastJob"]);
  renderStatus(local.lastStatus);
  renderJob(local.lastJob);
  await loadModels();

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", async () => {
      const mode = button.dataset.mode;
      button.disabled = true;
      statusBox.className = "status-box busy";
      statusMsg.textContent = `${modeLabel(mode)}: sending current page…`;
      chrome.runtime.sendMessage({ type: "enqueueCurrent", mode }, (result) => {
        button.disabled = false;
        if (chrome.runtime.lastError) {
          statusBox.className = "status-box err";
          statusMsg.textContent = chrome.runtime.lastError.message;
          return;
        }
        if (!result?.ok) {
          statusBox.className = "status-box err";
          statusMsg.textContent = result?.error || "Could not queue this page";
        }
      });
    });
  });

  document.getElementById("dashboard").addEventListener("click", async () => {
    const url = apiUrlInput.value.replace(/\/$/, "") || "http://localhost:8000";
    await chrome.tabs.create({ url });
  });

  document.getElementById("refreshModels").addEventListener("click", loadModels);

  document.getElementById("save").addEventListener("click", async () => {
    const apiUrl = apiUrlInput.value.replace(/\/$/, "");
    const apiSecret = apiSecretInput.value;
    await chrome.storage.sync.set({ apiUrl, apiSecret });
    try {
      const res = await fetch(`${apiUrl}/api/models`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiSecret}`,
        },
        body: JSON.stringify({
          categorize_model: catModel.value,
          analyze_model: anModel.value,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(formatDetail(data.detail) || "Model save failed");
      savedEl.textContent = `Saved · cat=${data.current.categorize_model} · an=${data.current.analyze_model}`;
    } catch (err) {
      savedEl.textContent = String(err.message || err);
      savedEl.style.color = "#dc2626";
      setTimeout(() => { savedEl.style.color = "#16a34a"; }, 3000);
    }
    setTimeout(() => { savedEl.textContent = ""; }, 4000);
  });

  document.getElementById("clear").addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "clearStatus" }, () => {
      statusBox.className = "status-box ok";
      statusMsg.textContent = "Ready";
    });
  });

  document.getElementById("batch").addEventListener("click", async () => {
    const config = {
      apiUrl: apiUrlInput.value.replace(/\/$/, ""),
      apiSecret: apiSecretInput.value,
    };
    const tabs = await chrome.tabs.query({ currentWindow: true });
    const items = tabs
      .filter((t) => t.url && /^https?:/i.test(t.url))
      .map((t) => ({ url: t.url, title: t.title || "" }));
    if (!items.length) {
      statusMsg.textContent = "No tabs to add";
      return;
    }
    statusBox.className = "status-box busy";
    statusMsg.textContent = `Batching ${items.length} tabs…`;
    try {
      const response = await fetch(`${config.apiUrl}/api/items/batch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.apiSecret}`,
        },
        body: JSON.stringify({ items, mode: "categorize" }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Batch failed");
      statusBox.className = "status-box ok";
      statusMsg.textContent = `Queued ${data.count} tabs`;
    } catch (err) {
      statusBox.className = "status-box err";
      statusMsg.textContent = String(err.message || err);
    }
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    if (changes.lastStatus) renderStatus(changes.lastStatus.newValue);
    if (changes.lastJob) renderJob(changes.lastJob.newValue);
  });

  setInterval(async () => {
    const { lastJob } = await chrome.storage.local.get("lastJob");
    if (!lastJob?.id || ["done", "error"].includes(lastJob.status)) return;
    try {
      const res = await fetch(`${apiUrlInput.value.replace(/\/$/, "")}/api/jobs/${lastJob.id}`);
      if (!res.ok) return;
      const job = await res.json();
      await chrome.storage.local.set({ lastJob: job });
      renderJob(job);
    } catch {
      // ignore
    }
  }, 2000);

  async function loadModels() {
    const apiUrl = apiUrlInput.value.replace(/\/$/, "") || "http://localhost:8000";
    try {
      const data = await fetch(`${apiUrl}/api/models`).then((r) => r.json());
      const chat = data.chat_models || data.installed || [];
      const cur = data.current || {};
      fillSelect(catModel, chat, cur.categorize_model);
      fillSelect(anModel, chat, cur.analyze_model);
      savedEl.textContent = chat.length ? `${chat.length} chat models found` : "No models — is Ollama running?";
      setTimeout(() => { if (savedEl.textContent.includes("found")) savedEl.textContent = ""; }, 2000);
    } catch {
      fillSelect(catModel, ["qwen3:4b"], "qwen3:4b");
      fillSelect(anModel, ["qwen3:8b"], "qwen3:8b");
      savedEl.textContent = "Could not reach /api/models";
    }
  }

  function fillSelect(el, options, selected) {
    el.innerHTML = options
      .map((o) => `<option value="${o}" ${o === selected ? "selected" : ""}>${o}</option>`)
      .join("");
    if (selected && ![...el.options].some((o) => o.value === selected)) {
      el.insertAdjacentHTML("afterbegin", `<option value="${selected}" selected>${selected}</option>`);
    }
  }

  function formatDetail(detail) {
    if (!detail) return "";
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    return JSON.stringify(detail);
  }

  function renderStatus(status) {
    if (!status) return;
    statusBox.className = `status-box ${status.kind || ""}`;
    statusMsg.textContent = status.message || "";
  }

  function renderJob(job) {
    if (!job) {
      jobBox.textContent = "No active job details yet.";
      return;
    }
    const title = escapeHtml(job.title || "Learning item");
    const state = escapeHtml(job.stage || job.status || "Queued");
    const details = [
      job.reused_page ? "Updated existing page" : null,
      job.category ? escapeHtml(job.category) : null,
      job.timings?.total != null ? `${job.timings.total}s` : null,
    ].filter(Boolean).join(" · ");
    const notion = job.notion_url
      ? `<a href="${escapeHtml(job.notion_url)}" target="_blank">Open in Notion ↗</a>`
      : "";
    jobBox.innerHTML = `<strong>${title}</strong><div>${state}${details ? ` · ${details}` : ""}</div>${notion}`;
  }

  function modeLabel(mode) {
    return ({ categorize: "Save", summarize: "Summarize", feynman: "Explain", paper: "Research paper" })[mode] || mode;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
  }
});
