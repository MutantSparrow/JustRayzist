const endpointListEl = document.getElementById("endpoint-list");
const baseUrlEl = document.getElementById("base-url");
const methodEl = document.getElementById("tester-method");
const pathEl = document.getElementById("tester-path");
const clientIdEl = document.getElementById("tester-client-id");
const bodyTypeEl = document.getElementById("tester-body-type");
const bodyEl = document.getElementById("tester-body");
const fileRowEl = document.getElementById("tester-file-row");
const fileLabelEl = document.getElementById("tester-file-label");
const fileInputEl = document.getElementById("tester-file-input");
const sendEl = document.getElementById("tester-send");
const clearEl = document.getElementById("tester-clear");
const statusEl = document.getElementById("tester-status");
const responseEl = document.getElementById("tester-response");
const deleteGalleryEl = document.getElementById("maintenance-delete-gallery");
const killServerEl = document.getElementById("maintenance-kill-server");
const disconnectOverlayEl = document.getElementById("disconnect-overlay");

const CLIENT_ID_STORAGE_KEY = "justrayzist.client_id";
const API_MANIFEST_PATH = "/api-manifest";
let endpoints = [];

function asJson(value) {
  return JSON.stringify(value, null, 2);
}

function safePath(input) {
  const raw = String(input || "").trim();
  if (!raw) return "/";
  return raw.startsWith("/") ? raw : `/${raw}`;
}

function setStatus(text, ok = true) {
  statusEl.textContent = text;
  statusEl.className = `status ${ok ? "ok" : "err"}`;
}

function createClientId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `client_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
}

function getOrCreateClientId() {
  let clientId = "";
  try {
    clientId = String(window.localStorage.getItem(CLIENT_ID_STORAGE_KEY) || "").trim();
  } catch (_) {
    clientId = "";
  }
  if (clientId) {
    return clientId;
  }
  clientId = createClientId();
  try {
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, clientId);
  } catch (_) {
  }
  return clientId;
}

function normalizeEndpoint(entry) {
  return {
    method: String(entry.method || "GET").toUpperCase(),
    path: String(entry.path || "/"),
    description: String(entry.description || ""),
    request: entry.request ?? null,
    response: entry.response ?? null,
    requiresClient: Boolean(entry.requires_client),
    requestMediaType: String(entry.request_media_type || "application/json"),
    fileFields: Array.isArray(entry.file_fields) ? entry.file_fields.map((item) => String(item || "").trim()).filter(Boolean) : [],
  };
}

function endpointSampleRequest(endpoint) {
  if (!endpoint || endpoint.request == null || typeof endpoint.request === "string") {
    return "{}";
  }
  const request = { ...endpoint.request };
  for (const fieldName of endpoint.fileFields || []) {
    delete request[fieldName];
  }
  return asJson(request);
}

function findEndpoint(method, path) {
  const targetMethod = String(method || "GET").toUpperCase();
  const targetPath = safePath(path);
  return endpoints.find((entry) => entry.method === targetMethod && entry.path === targetPath) || null;
}

function updateTesterMode(endpoint = null) {
  const currentEndpoint = endpoint || findEndpoint(methodEl.value, pathEl.value);
  const mediaType = currentEndpoint?.requestMediaType || "application/json";
  bodyTypeEl.value = mediaType;
  const fileField = currentEndpoint?.fileFields?.[0] || "";
  fileRowEl.classList.toggle("hidden", mediaType !== "multipart/form-data");
  fileLabelEl.textContent = fileField ? `Upload ${fileField}` : "Upload File";
  if (mediaType !== "multipart/form-data") {
    fileInputEl.value = "";
  }
}

function renderEndpoints() {
  endpointListEl.innerHTML = "";
  endpoints.forEach((endpoint) => {
    const card = document.createElement("article");
    card.className = "endpoint-card";

    const route = document.createElement("div");
    route.className = "route";
    route.innerHTML = `<span class="method ${endpoint.method}">${endpoint.method}</span><code>${endpoint.path}</code>`;

    const description = document.createElement("div");
    description.className = "description";
    const scopeSuffix = endpoint.requiresClient ? " Requires X-JustRayzist-Client." : "";
    const mediaSuffix =
      endpoint.requestMediaType && endpoint.requestMediaType !== "application/json"
        ? ` Uses ${endpoint.requestMediaType}.`
        : "";
    description.textContent = `${endpoint.description}${scopeSuffix}${mediaSuffix}`;

    const requestPre = document.createElement("pre");
    requestPre.textContent =
      endpoint.request == null ? "(no body)" : typeof endpoint.request === "string" ? endpoint.request : asJson(endpoint.request);

    const responsePre = document.createElement("pre");
    responsePre.textContent =
      typeof endpoint.response === "string" ? endpoint.response : asJson(endpoint.response);

    const useButton = document.createElement("button");
    useButton.type = "button";
    useButton.className = "fill-btn";
    useButton.textContent = "Use In Tester";
    useButton.addEventListener("click", () => {
      methodEl.value = endpoint.method;
      pathEl.value = endpoint.path;
      bodyEl.value = endpointSampleRequest(endpoint);
      updateTesterMode(endpoint);
    });

    card.append(route, description, useButton);

    const requestLabel = document.createElement("div");
    requestLabel.className = "description";
    requestLabel.textContent =
      endpoint.requestMediaType === "multipart/form-data"
        ? "Sample multipart fields"
        : "Sample request body";
    card.append(requestLabel, requestPre);

    const responseLabel = document.createElement("div");
    responseLabel.className = "description";
    responseLabel.textContent = "Sample response";
    card.append(responseLabel, responsePre);

    endpointListEl.append(card);
  });
}

async function sendRequest() {
  const method = String(methodEl.value || "GET").toUpperCase();
  const path = safePath(pathEl.value);
  const hasPlaceholders = path.includes("{") || path.includes("}");
  if (hasPlaceholders) {
    setStatus("Replace path placeholders before sending (for example: /images/my_file.png).", false);
    return;
  }

  const endpoint = findEndpoint(method, path);
  const mediaType = endpoint?.requestMediaType || "application/json";
  const clientId = String(clientIdEl.value || "").trim();
  const options = {
    method,
    headers: {},
  };
  if (clientId) {
    options.headers["X-JustRayzist-Client"] = clientId;
  }

  if (mediaType === "multipart/form-data") {
    const parsed = (() => {
      const raw = String(bodyEl.value || "").trim();
      if (!raw || raw === "{}") return {};
      try {
        return JSON.parse(raw);
      } catch (_) {
        throw new Error("Request body is not valid JSON.");
      }
    })();
    const body = new FormData();
    Object.entries(parsed).forEach(([key, value]) => {
      if (value === null || value === undefined || value === "") return;
      body.append(key, typeof value === "object" ? JSON.stringify(value) : String(value));
    });
    for (const fieldName of endpoint?.fileFields || []) {
      const file = fileInputEl.files?.[0] || null;
      if (!(file instanceof File)) {
        throw new Error(`Select a file for multipart field '${fieldName}'.`);
      }
      body.append(fieldName, file, file.name);
    }
    options.body = body;
  } else if (method === "POST" || method === "DELETE" || method === "PATCH") {
    const raw = String(bodyEl.value || "").trim();
    if (raw && raw !== "{}") {
      try {
        const parsed = JSON.parse(raw);
        options.body = JSON.stringify(parsed);
      } catch (_) {
        setStatus("Request body is not valid JSON.", false);
        return;
      }
    }
    options.headers["Content-Type"] = "application/json";
  }

  setStatus(`Sending ${method} ${path} ...`, true);
  responseEl.textContent = "";
  try {
    const response = await fetch(path, options);
    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");
    const payload = isJson ? await response.json() : await response.text();

    setStatus(`HTTP ${response.status} ${response.statusText}`, response.ok);
    responseEl.textContent = typeof payload === "string" ? payload : asJson(payload);
  } catch (error) {
    setStatus(`Request failed: ${String(error?.message || error)}`, false);
    responseEl.textContent = "";
  }
}

async function fetchEndpointManifest() {
  const response = await fetch(API_MANIFEST_PATH, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load API manifest (${response.status} ${response.statusText}).`);
  }
  const payload = await response.json();
  const items = Array.isArray(payload?.items) ? payload.items : [];
  endpoints = items.map(normalizeEndpoint);
}

function clearTester() {
  methodEl.value = "GET";
  pathEl.value = "/health";
  bodyEl.value = "{}";
  fileInputEl.value = "";
  statusEl.textContent = "";
  statusEl.className = "status";
  responseEl.textContent = "";
  updateTesterMode();
}

function startDisconnectEffect() {
  disconnectOverlayEl.classList.remove("active");
  disconnectOverlayEl.classList.remove("hidden");
  disconnectOverlayEl.setAttribute("aria-hidden", "false");
  void disconnectOverlayEl.offsetWidth;
  requestAnimationFrame(() => {
    disconnectOverlayEl.classList.add("active");
  });
}

function stopDisconnectEffect() {
  disconnectOverlayEl.classList.remove("active");
  disconnectOverlayEl.classList.add("hidden");
  disconnectOverlayEl.setAttribute("aria-hidden", "true");
}

async function deleteGallery() {
  const confirmation = window.prompt("Type DELETE to confirm gallery deletion for this client:");
  if (confirmation === null) return;
  const encoded = encodeURIComponent(confirmation);
  const clientId = String(clientIdEl.value || "").trim();
  const headers = { "Content-Type": "application/json" };
  if (clientId) {
    headers["X-JustRayzist-Client"] = clientId;
  }
  setStatus("Deleting gallery ...", true);
  responseEl.textContent = "";
  const response = await fetch(`/gallery?confirm=${encoded}`, {
    method: "DELETE",
    headers,
    body: JSON.stringify({ confirm: confirmation }),
  });
  const payload = await response.json();
  setStatus(`HTTP ${response.status} ${response.statusText}`, response.ok);
  responseEl.textContent = asJson(payload);
}

async function killServer() {
  if (!window.confirm("Kill the local server now?")) return;
  startDisconnectEffect();
  setStatus("Killing server ...", true);
  responseEl.textContent = "";
  try {
    const response = await fetch("/server/kill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      stopDisconnectEffect();
      setStatus(`HTTP ${response.status} ${response.statusText}`, false);
      responseEl.textContent = payload == null ? "" : asJson(payload);
      return;
    }
    setStatus(`HTTP ${response.status} ${response.statusText}`, true);
    responseEl.textContent = payload == null ? "" : asJson(payload);
  } catch (error) {
    if (error instanceof TypeError) {
      return;
    }
    stopDisconnectEffect();
    setStatus(`Request failed: ${String(error?.message || error)}`, false);
    responseEl.textContent = "";
  }
}

async function bootstrap() {
  baseUrlEl.textContent = window.location.origin;
  clientIdEl.value = getOrCreateClientId();
  await fetchEndpointManifest();
  renderEndpoints();
  clearTester();
}

sendEl.addEventListener("click", () => {
  sendRequest().catch((error) => {
    setStatus(`Request failed: ${String(error?.message || error)}`, false);
  });
});

clearEl.addEventListener("click", clearTester);
methodEl.addEventListener("change", () => updateTesterMode());
pathEl.addEventListener("input", () => updateTesterMode());
deleteGalleryEl.addEventListener("click", () => {
  deleteGallery().catch((error) => {
    setStatus(`Request failed: ${String(error?.message || error)}`, false);
  });
});
killServerEl.addEventListener("click", () => {
  killServer().catch((error) => {
    setStatus(`Request failed: ${String(error?.message || error)}`, false);
  });
});

bootstrap().catch((error) => {
  setStatus(`Failed to load API examples: ${String(error?.message || error)}`, false);
});
