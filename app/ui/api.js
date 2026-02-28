const endpointListEl = document.getElementById("endpoint-list");
const baseUrlEl = document.getElementById("base-url");
const methodEl = document.getElementById("tester-method");
const pathEl = document.getElementById("tester-path");
const bodyEl = document.getElementById("tester-body");
const sendEl = document.getElementById("tester-send");
const clearEl = document.getElementById("tester-clear");
const statusEl = document.getElementById("tester-status");
const responseEl = document.getElementById("tester-response");

const ENDPOINTS = [
  {
    method: "GET",
    path: "/health",
    description: "Service health and active runtime profile.",
    request: null,
    response: {
      status: "ok",
      app: "JustRayzist",
      version: "0.1.0",
      profile: "balanced",
      offline_mode: true,
    },
  },
  {
    method: "GET",
    path: "/config",
    description: "Resolved runtime configuration and paths.",
    request: null,
    response: { app_name: "JustRayzist", runtime_profile: { name: "balanced" } },
  },
  {
    method: "GET",
    path: "/model-packs",
    description: "List discovered and valid model packs.",
    request: null,
    response: {
      count: 1,
      items: [{ name: "Rayzist_bf16", architecture: "z_image_turbo" }],
    },
  },
  {
    method: "POST",
    path: "/generate",
    description: "Generate image from prompt and dimensions.",
    request: {
      prompt: "A cinematic skyline at sunrise",
      width: 1024,
      height: 1024,
      pack: "Rayzist_bf16",
      seed: 123456,
      scheduler_mode: "euler",
      enhance_prompt: false,
    },
    response: {
      filename: "justrayzist_YYYYMMDD_hhmmss_000.png",
      width: 1024,
      height: 1024,
      duration_ms: 12345,
      url: "/images/justrayzist_YYYYMMDD_hhmmss_000.png",
    },
  },
  {
    method: "POST",
    path: "/upscale",
    description: "Upscale + refine an existing gallery image by filename.",
    request: {
      filename: "justrayzist_YYYYMMDD_hhmmss_000.png",
      pack: "Rayzist_bf16",
      seed: 123456,
      scheduler_mode: "euler",
      enhance_prompt: false,
    },
    response: {
      filename: "justrayzist_YYYYMMDD_hhmmss_001.png",
      mode: "api_upscale",
      source_filename: "justrayzist_YYYYMMDD_hhmmss_000.png",
      duration_ms: 23456,
      url: "/images/justrayzist_YYYYMMDD_hhmmss_001.png",
    },
  },
  {
    method: "GET",
    path: "/images?prompt=skyline&limit=50&offset=0&newest_first=true",
    description: "List indexed images with optional filtering/paging.",
    request: null,
    response: {
      count: 1,
      limit: 50,
      offset: 0,
      items: [{ filename: "justrayzist_YYYYMMDD_hhmmss_000.png" }],
    },
  },
  {
    method: "GET",
    path: "/images/{filename}",
    description: "Download image file by filename.",
    request: null,
    response: "PNG binary response",
  },
  {
    method: "DELETE",
    path: "/images/{filename}?confirm=DELETE",
    description: "Delete one image and its index entry.",
    request: { confirm: "DELETE" },
    response: { status: "ok", deleted_files: 1, deleted_rows: 1, filename: "..." },
  },
  {
    method: "DELETE",
    path: "/gallery?confirm=DELETE",
    description: "Delete all gallery images and index entries.",
    request: { confirm: "DELETE" },
    response: { status: "ok", deleted_files: 42, deleted_rows: 42, remaining_rows: 0 },
  },
  {
    method: "POST",
    path: "/server/kill",
    description: "Request local server shutdown.",
    request: {},
    response: { status: "ok", message: "Server shutdown initiated." },
  },
];

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

function renderEndpoints() {
  endpointListEl.innerHTML = "";
  ENDPOINTS.forEach((endpoint) => {
    const card = document.createElement("article");
    card.className = "endpoint-card";

    const route = document.createElement("div");
    route.className = "route";
    route.innerHTML = `<span class="method ${endpoint.method}">${endpoint.method}</span><code>${endpoint.path}</code>`;

    const description = document.createElement("div");
    description.className = "description";
    description.textContent = endpoint.description;

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

  const options = { method, headers: {} };
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

function clearTester() {
  methodEl.value = "POST";
  pathEl.value = "/health";
  bodyEl.value = "{}";
  statusEl.textContent = "";
  statusEl.className = "status";
  responseEl.textContent = "";
}

function bootstrap() {
  baseUrlEl.textContent = window.location.origin;
  renderEndpoints();
  clearTester();
}

sendEl.addEventListener("click", () => {
  sendRequest().catch((error) => {
    setStatus(`Request failed: ${String(error?.message || error)}`, false);
  });
});

clearEl.addEventListener("click", clearTester);

bootstrap();
