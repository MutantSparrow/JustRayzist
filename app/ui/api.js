const endpointListEl = document.getElementById("endpoint-list");
const baseUrlEl = document.getElementById("base-url");
const methodEl = document.getElementById("tester-method");
const pathEl = document.getElementById("tester-path");
const clientIdEl = document.getElementById("tester-client-id");
const bodyEl = document.getElementById("tester-body");
const sendEl = document.getElementById("tester-send");
const clearEl = document.getElementById("tester-clear");
const statusEl = document.getElementById("tester-status");
const responseEl = document.getElementById("tester-response");

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
    description.textContent = `${endpoint.description}${scopeSuffix}`;

    const requestPre = document.createElement("pre");
    requestPre.textContent = endpoint.request == null ? "(no body)" : asJson(endpoint.request);

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
      bodyEl.value = endpoint.request == null ? "{}" : asJson(endpoint.request);
    });

    card.append(route, description, useButton);

    const requestLabel = document.createElement("div");
    requestLabel.className = "description";
    requestLabel.textContent = "Sample request body";
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

  const clientId = String(clientIdEl.value || "").trim();
  const options = {
    method,
    headers: {},
  };
  if (clientId) {
    options.headers["X-JustRayzist-Client"] = clientId;
  }

  if (method === "POST" || method === "DELETE") {
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
  endpoints = items.map((entry) => ({
    method: String(entry.method || "GET").toUpperCase(),
    path: String(entry.path || "/"),
    description: String(entry.description || ""),
    request: entry.request ?? null,
    response: entry.response ?? null,
    requiresClient: Boolean(entry.requires_client),
  }));
}

function clearTester() {
  methodEl.value = "GET";
  pathEl.value = "/health";
  bodyEl.value = "{}";
  statusEl.textContent = "";
  statusEl.className = "status";
  responseEl.textContent = "";
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

bootstrap().catch((error) => {
  setStatus(`Failed to load API examples: ${String(error?.message || error)}`, false);
});
