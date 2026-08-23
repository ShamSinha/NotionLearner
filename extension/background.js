const DEFAULT_API_URL = "http://localhost:8000";

chrome.runtime.onInstalled.addListener(setupMenus);
chrome.runtime.onStartup.addListener(setupMenus);
setupMenus();
clearStaleBusy();

function setupMenus() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "add-to-queue",
      title: "Add to Learning Queue",
      contexts: ["page", "link", "selection"],
    });
    chrome.contextMenus.create({
      id: "add-and-summarize",
      title: "Add & Summarize",
      contexts: ["page", "link", "selection"],
    });
    chrome.contextMenus.create({
      id: "add-feynman",
      title: "Add & Explain (Feynman)",
      contexts: ["page", "link", "selection"],
    });
    chrome.contextMenus.create({
      id: "add-paper",
      title: "Add as Research Paper",
      contexts: ["page", "link", "selection"],
    });
    chrome.contextMenus.create({
      id: "add-all-tabs",
      title: "Add ALL open tabs to Queue",
      contexts: ["action", "page"],
    });
  });
}

async function clearStaleBusy() {
  const stored = await chrome.storage.local.get("lastStatus");
  if (stored.lastStatus?.kind === "busy") {
    await setStatus("err", "Previous job interrupted — try again (jobs are async now)");
  }
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const config = await getConfig();
  if (!config.apiUrl || !config.apiSecret) {
    await setStatus("err", "Save Backend URL + API Secret in the popup first");
    return;
  }

  if (info.menuItemId === "add-all-tabs") {
    await addAllTabs(config);
    return;
  }

  const modeMap = {
    "add-to-queue": "categorize",
    "add-and-summarize": "summarize",
    "add-feynman": "feynman",
    "add-paper": "paper",
  };
  const mode = modeMap[info.menuItemId] || "categorize";
  const url = info.linkUrl || info.pageUrl || tab?.url;
  if (!url) return;

  await enqueueOne(config, {
    url,
    title: tab?.title || "",
    page_html: await grabHtml(tab, url),
    selected_text: info.selectionText || null,
    mode,
  });
});

async function addAllTabs(config) {
  const tabs = await chrome.tabs.query({ currentWindow: true });
  const items = tabs
    .filter((t) => t.url && /^https?:/i.test(t.url))
    .map((t) => ({ url: t.url, title: t.title || "" }));
  if (!items.length) {
    await setStatus("err", "No http(s) tabs to add");
    return;
  }
  await setStatus("busy", `Batching ${items.length} tabs…`);
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
    if (!response.ok) throw new Error(formatDetail(data.detail) || "Batch failed");
    await setStatus("ok", `Queued ${data.count} tabs — watch localhost:8000`);
    pollJobs(config, data.job_ids || []);
  } catch (err) {
    await setStatus("err", String(err.message || err));
  }
}

async function enqueueOne(config, payload) {
  try {
    const health = await fetch(`${config.apiUrl}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!health.ok) throw new Error("Backend down");
  } catch {
    await setStatus("err", `Cannot reach ${config.apiUrl}`);
    return { ok: false, error: `Cannot reach ${config.apiUrl}` };
  }

  await setStatus("busy", `${payload.mode}: accepted — processing in background…`);
  try {
    const response = await fetch(`${config.apiUrl}/api/items`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.apiSecret}`,
      },
      body: JSON.stringify({
        url: payload.url,
        title: payload.title,
        page_html: payload.page_html || "",
        selected_text: payload.selected_text,
        mode: payload.mode,
        summarize: payload.mode !== "categorize",
        async_mode: true,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(formatDetail(data.detail) || `HTTP ${response.status}`);
    await setStatus("busy", `Job ${data.job_id}: ${payload.mode} running…`);
    pollJobs(config, [data.job_id]);
    return { ok: true, jobId: data.job_id };
  } catch (err) {
    await setStatus("err", String(err.message || err));
    return { ok: false, error: String(err.message || err) };
  }
}

function pollJobs(config, jobIds) {
  const pending = new Set(jobIds);
  const timer = setInterval(async () => {
    try {
      for (const id of [...pending]) {
        const res = await fetch(`${config.apiUrl}/api/jobs/${id}`);
        if (!res.ok) continue;
        const job = await res.json();
        await chrome.storage.local.set({ lastJob: job });
        if (job.status === "done") {
          pending.delete(id);
          await setStatus(
            "ok",
            `${job.title} → ${job.category || "done"} (${job.timings?.total || "?"}s)`
          );
        } else if (job.status === "error") {
          pending.delete(id);
          await setStatus("err", job.error || "Job failed");
        } else {
          await setStatus("busy", `${job.stage}: ${job.message || job.status}`);
        }
      }
      if (pending.size === 0) clearInterval(timer);
    } catch {
      // keep polling briefly
    }
  }, 2000);
  // safety stop
  setTimeout(() => clearInterval(timer), 30 * 60 * 1000);
}

async function grabHtml(tab, url) {
  if (!tab?.id || isYouTube(url)) return "";
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.documentElement.outerHTML,
    });
    return result || "";
  } catch {
    return "";
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "clearStatus") {
    setStatus("ok", "Ready").then(() => {
      chrome.action.setBadgeText({ text: "" });
      sendResponse({ ok: true });
    });
    return true;
  }
  if (msg?.type === "getConfig") {
    getConfig().then(sendResponse);
    return true;
  }
  if (msg?.type === "enqueueCurrent") {
    (async () => {
      try {
        const config = await getConfig();
        if (!config.apiUrl || !config.apiSecret) {
          const error = "Save Backend URL + API Secret in Settings first";
          await setStatus("err", error);
          sendResponse({ ok: false, error });
          return;
        }
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const url = tab?.url || "";
        if (!/^https?:/i.test(url)) {
          sendResponse({ ok: false, error: "The current tab is not an HTTP(S) page" });
          return;
        }
        const mode = ["categorize", "summarize", "feynman", "paper"].includes(msg.mode)
          ? msg.mode
          : "categorize";
        const result = await enqueueOne(config, {
          url,
          title: tab.title || "",
          page_html: await grabHtml(tab, url),
          selected_text: null,
          mode,
        });
        sendResponse(result);
      } catch (err) {
        sendResponse({ ok: false, error: String(err.message || err) });
      }
    })();
    return true;
  }
});

function formatDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  return JSON.stringify(detail);
}

async function setStatus(kind, message) {
  const colors = { busy: "#2563eb", ok: "#16a34a", err: "#dc2626" };
  const texts = { busy: "…", ok: "✓", err: "!" };
  await chrome.action.setBadgeBackgroundColor({ color: colors[kind] || "#666" });
  await chrome.action.setBadgeText({ text: texts[kind] || "" });
  await chrome.action.setTitle({ title: `NotionLearner\n${message}` });
  await chrome.storage.local.set({ lastStatus: { kind, message, at: Date.now() } });
  if (kind === "ok") {
    setTimeout(async () => {
      const stored = await chrome.storage.local.get("lastStatus");
      if (stored.lastStatus?.kind === "ok") await chrome.action.setBadgeText({ text: "" });
    }, 5000);
  }
}

function isYouTube(url) {
  return /youtube\.com|youtu\.be/.test(url);
}

async function getConfig() {
  const stored = await chrome.storage.sync.get(["apiUrl", "apiSecret"]);
  return {
    apiUrl: stored.apiUrl || DEFAULT_API_URL,
    apiSecret: stored.apiSecret || "",
  };
}
