const topbarEl = document.querySelector(".topbar");
const promptInputEl = document.getElementById("prompt-input");
const generateButtonEl = document.getElementById("generate-button");
const generateButtonLabelEl = document.getElementById("generate-button-label");
const settingsButtonEl = document.getElementById("settings-button");
const settingsPanelEl = document.getElementById("settings-panel");
const settingsSummaryEl = document.getElementById("settings-summary");
const resolutionSelectEl = document.getElementById("resolution-select");
const orientationToggleEl = document.getElementById("orientation-toggle");
const freezeSeedButtonEl = document.getElementById("freeze-seed-button");
const proceduralLatentSliderEl = document.getElementById("procedural-latent-slider");
const proceduralLatentValueEl = document.getElementById("procedural-latent-value");
const promptEnhanceButtonEl = document.getElementById("prompt-enhance-button");
const deleteGalleryButtonEl = document.getElementById("delete-gallery-button");
const killServerButtonEl = document.getElementById("kill-server-button");
const filterInputEl = document.getElementById("filter-input");
const reverseOrderButtonEl = document.getElementById("reverse-order-button");
const galleryColorFiltersEl = document.getElementById("gallery-color-filters");
const galleryDensitySliderEl = document.getElementById("gallery-density-slider");
const multiSelectToggleEl = document.getElementById("multi-select-toggle");
const multiSelectActionsEl = document.getElementById("multi-select-actions");
const multiSelectCountEl = document.getElementById("multi-select-count");
const bulkUpscaleButtonEl = document.getElementById("bulk-upscale-button");
const bulkDownloadButtonEl = document.getElementById("bulk-download-button");
const bulkDeleteButtonEl = document.getElementById("bulk-delete-button");
const galleryCountEl = document.getElementById("gallery-count");
const statusLineEl = document.getElementById("status-line");
const galleryEl = document.getElementById("gallery");
const emptyStateEl = document.getElementById("empty-state");
const viewerModalEl = document.getElementById("viewer-modal");
const viewerMetaEl = document.getElementById("viewer-meta");
const viewerImageEl = document.getElementById("viewer-image");
const viewerDownloadEl = document.getElementById("viewer-download");
const viewerStageEl = document.getElementById("viewer-stage");
const viewerCloseButtonEl = document.getElementById("viewer-close-button");
const viewerUsePromptButtonEl = document.getElementById("viewer-use-prompt-button");
const viewerCopyPromptButtonEl = document.getElementById("viewer-copy-prompt-button");
const viewerUpscaleButtonEl = document.getElementById("viewer-upscale-button");
const viewerPrevButtonEl = document.getElementById("viewer-prev-button");
const viewerNextButtonEl = document.getElementById("viewer-next-button");
const zoomLabelEl = document.getElementById("zoom-label");
const confirmModalEl = document.getElementById("confirm-modal");
const confirmModalMessageEl = document.getElementById("confirm-modal-message");
const confirmModalCancelEl = document.getElementById("confirm-modal-cancel");
const confirmModalConfirmEl = document.getElementById("confirm-modal-confirm");
const zipProgressModalEl = document.getElementById("zip-progress-modal");
const zipProgressCancelEl = document.getElementById("zip-progress-cancel");
const disconnectOverlayEl = document.getElementById("disconnect-overlay");

const requiredUi = [
  ["topbar", topbarEl],
  ["prompt-input", promptInputEl],
  ["generate-button", generateButtonEl],
  ["generate-button-label", generateButtonLabelEl],
  ["settings-button", settingsButtonEl],
  ["settings-panel", settingsPanelEl],
  ["settings-summary", settingsSummaryEl],
  ["resolution-select", resolutionSelectEl],
  ["orientation-toggle", orientationToggleEl],
  ["freeze-seed-button", freezeSeedButtonEl],
  ["procedural-latent-slider", proceduralLatentSliderEl],
  ["procedural-latent-value", proceduralLatentValueEl],
  ["prompt-enhance-button", promptEnhanceButtonEl],
  ["delete-gallery-button", deleteGalleryButtonEl],
  ["kill-server-button", killServerButtonEl],
  ["filter-input", filterInputEl],
  ["reverse-order-button", reverseOrderButtonEl],
  ["gallery-color-filters", galleryColorFiltersEl],
  ["gallery-density-slider", galleryDensitySliderEl],
  ["multi-select-toggle", multiSelectToggleEl],
  ["multi-select-actions", multiSelectActionsEl],
  ["multi-select-count", multiSelectCountEl],
  ["bulk-upscale-button", bulkUpscaleButtonEl],
  ["bulk-download-button", bulkDownloadButtonEl],
  ["bulk-delete-button", bulkDeleteButtonEl],
  ["gallery-count", galleryCountEl],
  ["status-line", statusLineEl],
  ["gallery", galleryEl],
  ["empty-state", emptyStateEl],
  ["viewer-modal", viewerModalEl],
  ["viewer-meta", viewerMetaEl],
  ["viewer-image", viewerImageEl],
  ["viewer-download", viewerDownloadEl],
  ["viewer-stage", viewerStageEl],
  ["viewer-close-button", viewerCloseButtonEl],
  ["viewer-use-prompt-button", viewerUsePromptButtonEl],
  ["viewer-copy-prompt-button", viewerCopyPromptButtonEl],
  ["viewer-upscale-button", viewerUpscaleButtonEl],
  ["viewer-prev-button", viewerPrevButtonEl],
  ["viewer-next-button", viewerNextButtonEl],
  ["zoom-label", zoomLabelEl],
  ["confirm-modal", confirmModalEl],
  ["confirm-modal-message", confirmModalMessageEl],
  ["confirm-modal-cancel", confirmModalCancelEl],
  ["confirm-modal-confirm", confirmModalConfirmEl],
  ["zip-progress-modal", zipProgressModalEl],
  ["zip-progress-cancel", zipProgressCancelEl],
  ["disconnect-overlay", disconnectOverlayEl],
];

const missingUi = requiredUi.filter(([, element]) => !element).map(([name]) => name);
if (missingUi.length) {
  throw new Error(`UI initialization failed. Missing element(s): ${missingUi.join(", ")}`);
}

const CLIENT_ID_STORAGE_KEY = "justrayzist.client_id";
const GALLERY_COLUMNS_STORAGE_KEY = "justrayzist.gallery_columns";
const CLIENT_QUEUE_STORAGE_KEY = "justrayzist.client_queue";
const CLIENT_JOB_POLL_INTERVAL_MS = 1500;
const CLIENT_QUEUE_STORAGE_VERSION = 2;
const GALLERY_COLOR_FILTERS = ["black", "white", "red", "yellow", "blue", "green"];
const GALLERY_COLOR_CACHE_STATUS_MESSAGE = "Updating gallery color cache...";
const GALLERY_COLOR_CACHE_POLL_INTERVAL_MS = 2500;

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

function getStoredGalleryColumns() {
  try {
    const raw = Number(window.localStorage.getItem(GALLERY_COLUMNS_STORAGE_KEY) || 4);
    return Math.max(3, Math.min(8, raw || 4));
  } catch (_) {
    return 4;
  }
}

const state = {
  clientId: getOrCreateClientId(),
  orientation: "portrait",
  freezeSeed: false,
  proceduralCreativity: 0,
  promptEnhance: true,
  galleryColumns: getStoredGalleryColumns(),
  currentSeed: null,
  newestFirst: true,
  activeColorFilter: null,
  filterTimer: null,
  maxQueuedGenerations: 5,
  queue: [],
  activeJob: null,
  queueWorkerRunning: false,
  galleryItems: [],
  multiSelectMode: false,
  selectedFilenames: new Set(),
  pendingJobs: [],
  galleryLoadRequestSeq: 0,
  zoom: 1.0,
  panX: 0,
  panY: 0,
  dragging: false,
  dragStartX: 0,
  dragStartY: 0,
  dragBaseX: 0,
  dragBaseY: 0,
  viewerIndex: -1,
  viewerFilename: null,
  viewerPromptExpanded: false,
  viewerCompareHolding: false,
  viewerCompareSourceFilename: null,
  confirmAction: null,
  zipAbortController: null,
  galleryColumnFrame: null,
  pendingGalleryColumns: 4,
  galleryRelayoutFrame: null,
  clientJobPollTimer: null,
  galleryColorCacheActive: false,
  galleryColorCachePollTimer: null,
};

function randomSeed() {
  return Math.floor(Math.random() * 2_147_483_646) + 1;
}

function updateTopbarOffset() {
  const offset = topbarEl.offsetHeight;
  const topbarRect = topbarEl.getBoundingClientRect();
  const promptRect = promptInputEl.getBoundingClientRect();
  const promptTop = Math.max(0, Math.round(promptRect.top - topbarRect.top));
  const buttonTop = Math.max(0, Math.round(generateButtonEl.offsetTop));
  const generateShift = window.innerWidth <= 960 ? 0 : promptTop - buttonTop;
  document.documentElement.style.setProperty("--topbar-offset", `${offset}px`);
  document.documentElement.style.setProperty("--generate-shift", `${generateShift}px`);
  if (isSettingsOpen()) {
    positionSettingsPanel();
  }
}

function setStatus(message, isError = false) {
  statusLineEl.textContent = String(message || "");
  statusLineEl.classList.toggle("error", Boolean(isError));
}

function clearGalleryColorCachePoll() {
  if (!state.galleryColorCachePollTimer) return;
  window.clearTimeout(state.galleryColorCachePollTimer);
  state.galleryColorCachePollTimer = null;
}

function syncGalleryColorCacheStatusLine() {
  if (state.queue.length > 0 || state.activeJob) {
    return;
  }
  if (state.galleryColorCacheActive) {
    if (!statusLineEl.classList.contains("error")) {
      setStatus(GALLERY_COLOR_CACHE_STATUS_MESSAGE);
    }
    return;
  }
  if (statusLineEl.textContent === GALLERY_COLOR_CACHE_STATUS_MESSAGE && !statusLineEl.classList.contains("error")) {
    setStatus("Ready.");
  }
}

function scheduleGalleryColorCachePoll() {
  clearGalleryColorCachePoll();
  if (!state.galleryColorCacheActive) {
    return;
  }
  state.galleryColorCachePollTimer = window.setTimeout(() => {
    loadImages().catch((error) => setStatus(String(error?.message || error), true));
  }, GALLERY_COLOR_CACHE_POLL_INTERVAL_MS);
}

function isSettingsOpen() {
  return settingsPanelEl.classList.contains("open");
}

function positionSettingsPanel() {
  const margin = 8;
  const gap = 6;
  const triggerRect = settingsButtonEl.getBoundingClientRect();
  const panelRect = settingsPanelEl.getBoundingClientRect();
  const panelWidth = panelRect.width || Math.min(420, Math.max(280, window.innerWidth - margin * 2));
  const panelHeight = panelRect.height || 0;

  let left = triggerRect.left;
  let top = triggerRect.top - panelHeight - gap;
  const maxLeft = Math.max(margin, window.innerWidth - panelWidth - margin);
  left = Math.min(Math.max(margin, left), maxLeft);

  if (top < margin) {
    const belowTop = triggerRect.bottom + gap;
    if (belowTop + panelHeight <= window.innerHeight - margin) {
      top = belowTop;
    } else {
      top = Math.max(margin, window.innerHeight - panelHeight - margin);
    }
  }

  settingsPanelEl.style.left = `${Math.round(left)}px`;
  settingsPanelEl.style.top = `${Math.round(top)}px`;
}

function formatApiError(payload, fallback = "Request failed.") {
  if (!payload) return fallback;
  if (typeof payload === "string") return payload;
  const detail = payload.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (!item || typeof item !== "object") return String(item);
        const where = Array.isArray(item.loc) ? item.loc.join(".") : "field";
        const message = item.msg || "Invalid value";
        return `${where}: ${message}`;
      })
      .filter(Boolean);
    return parts.join(" | ") || fallback;
  }
  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch (_) {
      return fallback;
    }
  }
  return fallback;
}

function buildClientHeaders(existing) {
  const headers = new Headers(existing || {});
  headers.set("X-JustRayzist-Client", state.clientId);
  return headers;
}

async function apiFetch(path, options = {}) {
  const requestOptions = {
    ...options,
    headers: buildClientHeaders(options.headers),
  };
  return fetch(path, requestOptions);
}

function setSettingsVisible(visible) {
  settingsPanelEl.classList.toggle("open", visible);
  settingsPanelEl.setAttribute("aria-hidden", String(!visible));
  settingsButtonEl.setAttribute("aria-expanded", String(visible));
  if (visible) {
    positionSettingsPanel();
  }
}

function toggleSettingsVisible() {
  setSettingsVisible(!isSettingsOpen());
}

function parseResolution(value) {
  const chunks = String(value || "1024x1024").toLowerCase().split("x");
  const width = Number(chunks[0] || 1024);
  const height = Number(chunks[1] || 1024);
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    return { width: 1024, height: 1024 };
  }
  if (state.orientation === "landscape" && width !== height) {
    return { width: height, height: width };
  }
  return { width, height };
}

function parseTimestamp(raw) {
  if (!raw) return null;
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed;
}

function formatTimestamp(raw) {
  const parsed = parseTimestamp(raw);
  if (!parsed) return raw ? String(raw) : "Unknown date";
  return parsed.toLocaleString();
}

function shortPrompt(value, limit = 84) {
  const raw = String(value || "").trim();
  if (raw.length <= limit) return raw;
  return `${raw.slice(0, limit - 3)}...`;
}

function formatGalleryTimestamp(raw) {
  const parsed = parseTimestamp(raw);
  if (!parsed) return raw ? String(raw) : "Unknown date";
  const now = new Date();
  const dayStartNow = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const dayStartParsed = new Date(parsed.getFullYear(), parsed.getMonth(), parsed.getDate()).getTime();
  const dayDiff = (dayStartNow - dayStartParsed) / 86_400_000;
  const timeLabel = parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (dayDiff === 0) return `Today ${timeLabel}`;
  if (dayDiff === 1) return `Yesterday ${timeLabel}`;
  return parsed.toLocaleString();
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function buildImageUrl(filename) {
  const query = new URLSearchParams();
  query.set("client_id", state.clientId);
  return `/images/${encodeURIComponent(filename)}?${query.toString()}`;
}

function buildViewerImageUrl(filename) {
  const query = new URLSearchParams();
  query.set("client_id", state.clientId);
  query.set("t", String(Date.now()));
  return `/images/${encodeURIComponent(filename)}?${query.toString()}`;
}

function buildDownloadUrl(filename) {
  const query = new URLSearchParams();
  query.set("client_id", state.clientId);
  return `/images/${encodeURIComponent(filename)}?${query.toString()}`;
}

function sanitizeJobStatus(rawValue, kind = "generate") {
  const value = String(rawValue || "").trim().toLowerCase();
  if (value === "queued") return "queued";
  if (value === "cancelling") return "cancelling";
  if (value === "upscaling") return kind === "upscale" ? "upscaling" : "generating";
  return "generating";
}

function isActiveJobStatus(status) {
  return sanitizeJobStatus(status) !== "queued";
}

function normalizeJobTimestamp(rawValue) {
  const value = Number(rawValue);
  if (Number.isFinite(value) && value > 0) {
    return value;
  }
  return Date.now();
}

function sanitizeJobKind(rawValue) {
  return rawValue === "upscale" ? "upscale" : "generate";
}

function normalizeStoredJob(rawJob, overrides = {}) {
  if (!rawJob || typeof rawJob !== "object") return null;
  const placeholderId = String(
    overrides.placeholderId || rawJob.placeholderId || rawJob.job_id || "",
  ).trim();
  if (!placeholderId) return null;
  const kind = sanitizeJobKind(overrides.kind || rawJob.kind);
  const width = Math.max(1, Number(overrides.width ?? rawJob.width) || 1);
  const height = Math.max(1, Number(overrides.height ?? rawJob.height) || 1);
  return {
    kind,
    placeholderId,
    prompt: kind === "generate" ? String(overrides.prompt ?? rawJob.prompt ?? "").trim() : "",
    filename: kind === "upscale" ? String(overrides.filename ?? rawJob.filename ?? "").trim() : "",
    source_filename:
      kind === "upscale"
        ? String(overrides.source_filename ?? rawJob.source_filename ?? rawJob.filename ?? "").trim()
        : "",
    width,
    height,
    pack: rawJob.pack ?? null,
    seed: rawJob.seed ?? null,
    enhance_prompt: Boolean(overrides.enhance_prompt ?? rawJob.enhance_prompt),
    procedural_creativity: Number(overrides.procedural_creativity ?? rawJob.procedural_creativity ?? 0),
    status: sanitizeJobStatus(overrides.status || rawJob.status || "queued", kind),
    enqueuedAt: normalizeJobTimestamp(overrides.enqueuedAt ?? rawJob.enqueuedAt ?? rawJob.enqueued_at),
    queueIndex: Number(overrides.queueIndex ?? rawJob.queueIndex ?? 0),
    remoteInFlight: Boolean(overrides.remoteInFlight ?? rawJob.remoteInFlight),
  };
}

function serializeQueuedJob(job) {
  return {
    kind: sanitizeJobKind(job.kind),
    placeholderId: String(job.placeholderId || ""),
    prompt: String(job.prompt || ""),
    filename: String(job.filename || ""),
    source_filename: String(job.source_filename || ""),
    width: Math.max(1, Number(job.width) || 1),
    height: Math.max(1, Number(job.height) || 1),
    pack: job.pack ?? null,
    seed: job.seed ?? null,
    enhance_prompt: Boolean(job.enhance_prompt),
    procedural_creativity: Number(job.procedural_creativity || 0),
    status: sanitizeJobStatus(job.status, sanitizeJobKind(job.kind)),
    enqueuedAt: normalizeJobTimestamp(job.enqueuedAt),
    remoteInFlight: Boolean(job.remoteInFlight),
  };
}

function buildPendingSnapshot() {
  const items = [];
  if (state.activeJob) {
    items.push({
      ...serializeQueuedJob(state.activeJob),
      queueIndex: -1,
    });
  }
  state.queue.forEach((job, index) => {
    items.push({
      ...serializeQueuedJob({ ...job, status: "queued" }),
      queueIndex: index,
    });
  });
  return items;
}

function syncPendingJobsFromQueueState() {
  const next = [];
  if (state.activeJob) {
    next.push({
      ...serializeQueuedJob(state.activeJob),
      status: sanitizeJobStatus(state.activeJob.status, state.activeJob.kind),
    });
  }
  for (const job of state.queue) {
    next.push({
      ...serializeQueuedJob(job),
      status: "queued",
    });
  }
  state.pendingJobs = next;
}

function persistClientQueueState() {
  syncPendingJobsFromQueueState();
  try {
    const payload = {
      version: CLIENT_QUEUE_STORAGE_VERSION,
      pending_jobs: buildPendingSnapshot(),
    };
    window.localStorage.setItem(CLIENT_QUEUE_STORAGE_KEY, JSON.stringify(payload));
  } catch (_) {
  }
}

function pendingJobsFromStoredPayload(payload) {
  const pendingJobs = Array.isArray(payload?.pending_jobs) ? payload.pending_jobs : null;
  if (pendingJobs) {
    return pendingJobs;
  }
  const legacyJobs = [];
  const activeJob = payload?.active_job || null;
  if (activeJob) {
    legacyJobs.push({
      ...activeJob,
      status: "generating",
      queueIndex: -1,
    });
  }
  const queuedJobs = Array.isArray(payload?.queued_jobs) ? payload.queued_jobs : [];
  queuedJobs.forEach((job, index) => {
    legacyJobs.push({
      ...job,
      status: "queued",
      queueIndex: index,
    });
  });
  return legacyJobs;
}

function restoreClientQueueState() {
  state.queue = [];
  state.activeJob = null;
  syncPendingJobsFromQueueState();
  try {
    const raw = window.localStorage.getItem(CLIENT_QUEUE_STORAGE_KEY);
    if (!raw) return;
    const payload = JSON.parse(raw);
    const pendingJobs = pendingJobsFromStoredPayload(payload);
    const normalizedPendingJobs = pendingJobs
      .map((job, index) =>
        normalizeStoredJob(job, {
          status: isActiveJobStatus(job?.status) ? job?.status : "queued",
          remoteInFlight: isActiveJobStatus(job?.status),
          queueIndex: Number(job?.queueIndex ?? index),
        })
      )
      .filter(Boolean);
    const activeJobs = normalizedPendingJobs.filter((job) => isActiveJobStatus(job.status));
    state.activeJob = activeJobs.length > 0 ? activeJobs[0] : null;
    state.queue = normalizedPendingJobs
      .filter((job) => job.status !== "generating")
      .sort((left, right) => {
        const leftIndex = Number(left.queueIndex ?? Number.MAX_SAFE_INTEGER);
        const rightIndex = Number(right.queueIndex ?? Number.MAX_SAFE_INTEGER);
        if (leftIndex !== rightIndex) {
          return leftIndex - rightIndex;
        }
        return left.enqueuedAt - right.enqueuedAt;
      })
      .map((job) => ({
        ...job,
        status: "queued",
        remoteInFlight: false,
      }));
    syncPendingJobsFromQueueState();
  } catch (_) {
    state.queue = [];
    state.activeJob = null;
    syncPendingJobsFromQueueState();
  }
}

function stopClientJobPolling() {
  if (state.clientJobPollTimer === null) return;
  window.clearTimeout(state.clientJobPollTimer);
  state.clientJobPollTimer = null;
}

function scheduleClientJobPoll() {
  if (!state.activeJob || !state.activeJob.remoteInFlight) {
    stopClientJobPolling();
    return;
  }
  if (state.clientJobPollTimer !== null) {
    return;
  }
  state.clientJobPollTimer = window.setTimeout(async () => {
    state.clientJobPollTimer = null;
    try {
      await refreshClientJobState({ loadImagesOnRemoteCompletion: true });
    } catch (error) {
      setStatus(String(error?.message || error), true);
    } finally {
      if (state.activeJob && state.activeJob.remoteInFlight) {
        scheduleClientJobPoll();
      }
    }
  }, CLIENT_JOB_POLL_INTERVAL_MS);
}

async function refreshClientJobState(options = {}) {
  const loadImagesOnRemoteCompletion = Boolean(options.loadImagesOnRemoteCompletion);
  const response = await apiFetch("/client-jobs", { cache: "no-store" });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(formatApiError(payload, "Failed to load client job state."));
  }
  const remoteJob = normalizeStoredJob(payload?.active_job, {
    status: payload?.active_job?.status || "generating",
    remoteInFlight: true,
  });

  if (remoteJob) {
    const matchingQueuedJob =
      state.queue.find((job) => job.placeholderId === remoteJob.placeholderId) || null;
    const matchingActiveJob =
      state.activeJob && state.activeJob.placeholderId === remoteJob.placeholderId ? state.activeJob : null;
    state.queue = state.queue.filter((job) => job.placeholderId !== remoteJob.placeholderId);
    state.activeJob = normalizeStoredJob(
      {
        ...matchingQueuedJob,
        ...matchingActiveJob,
        ...remoteJob,
      },
      {
        placeholderId: remoteJob.placeholderId,
        status: remoteJob.status,
        remoteInFlight: true,
        enqueuedAt:
          matchingActiveJob?.enqueuedAt ??
          matchingQueuedJob?.enqueuedAt ??
          remoteJob.enqueuedAt,
      },
    );
    persistClientQueueState();
    renderGallery();
    updateGenerateButtonState();
    scheduleClientJobPoll();
    return;
  }

  stopClientJobPolling();
  if (state.activeJob && state.activeJob.remoteInFlight) {
    state.activeJob = null;
    persistClientQueueState();
    renderGallery();
    updateGenerateButtonState();
    if (loadImagesOnRemoteCompletion) {
      await loadImages();
    }
    processGenerationQueue().catch((error) => setStatus(String(error?.message || error), true));
  }
}

function galleryKeyForPending(job) {
  return `pending:${job.placeholderId}`;
}

function galleryKeyForImage(item) {
  return `image:${item.filename}`;
}

function getGalleryImageItem(filename) {
  const target = String(filename || "").trim();
  if (!target) return null;
  return state.galleryItems.find((item) => item.filename === target) || null;
}

function getGalleryNodeMap() {
  const map = new Map();
  const nodes = galleryEl.querySelectorAll("[data-gallery-key]");
  for (const node of nodes) {
    map.set(node.dataset.galleryKey, node);
  }
  return map;
}

function getResponsiveGalleryColumns(desired) {
  const next = Math.max(1, Number(desired) || 1);
  if (window.innerWidth <= 600) return 1;
  if (window.innerWidth <= 960) return Math.min(2, next);
  if (window.innerWidth <= 1280) return Math.min(3, next);
  return next;
}

function getGalleryGap() {
  const styles = window.getComputedStyle(galleryEl);
  const raw = parseFloat(styles.gap || styles.rowGap || "0");
  return Number.isFinite(raw) ? raw : 16;
}

function getTileAspectRatio(tile) {
  const width = Number(tile?.dataset?.aspectWidth || 0);
  const height = Number(tile?.dataset?.aspectHeight || 0);
  if (width > 0 && height > 0) {
    return width / height;
  }
  return 1;
}

function applyMasonryLayout() {
  const nodes = [...galleryEl.querySelectorAll("[data-gallery-key]")].filter(
    (node) => !node.classList.contains("removing"),
  );
  if (nodes.length === 0) {
    galleryEl.style.height = "";
    return;
  }

  const gap = getGalleryGap();
  const maxResponsiveColumns = getResponsiveGalleryColumns(state.galleryColumns);
  const containerWidth = Math.max(0, galleryEl.clientWidth);
  const maxFitColumns = Math.max(1, Math.floor((containerWidth + gap) / (180 + gap)) || 1);
  const columns = Math.max(1, Math.min(maxResponsiveColumns, maxFitColumns));
  const columnWidth = Math.max(1, Math.floor((containerWidth - gap * (columns - 1)) / columns));
  const heights = new Array(columns).fill(0);

  for (const node of nodes) {
    const ratio = getTileAspectRatio(node);
    const tileHeight = Math.max(120, Math.round(columnWidth / Math.max(0.1, ratio)));
    let columnIndex = 0;
    for (let index = 1; index < heights.length; index += 1) {
      if (heights[index] < heights[columnIndex]) {
        columnIndex = index;
      }
    }
    node.style.width = `${columnWidth}px`;
    node.style.height = `${tileHeight}px`;
    node.style.left = `${columnIndex * (columnWidth + gap)}px`;
    node.style.top = `${heights[columnIndex]}px`;
    heights[columnIndex] += tileHeight + gap;
  }

  galleryEl.style.height = `${Math.max(0, Math.max(...heights) - gap)}px`;
}

function applyGalleryLayout() {
  applyMasonryLayout();
}

function scheduleGalleryRelayout({ animate = false } = {}) {
  const before = animate ? captureGalleryPositions() : null;
  if (state.galleryRelayoutFrame !== null) {
    window.cancelAnimationFrame(state.galleryRelayoutFrame);
    state.galleryRelayoutFrame = null;
  }
  state.galleryRelayoutFrame = window.requestAnimationFrame(() => {
    state.galleryRelayoutFrame = null;
    applyGalleryLayout();
    if (before) {
      window.requestAnimationFrame(() => animateGalleryLayout(before));
    }
  });
}

function captureGalleryPositions() {
  const positions = new Map();
  const nodes = galleryEl.querySelectorAll("[data-gallery-key]");
  for (const node of nodes) {
    if (node.classList.contains("removing")) continue;
    positions.set(node.dataset.galleryKey, node.getBoundingClientRect());
  }
  return positions;
}

function animateGalleryLayout(beforePositions) {
  const nodes = galleryEl.querySelectorAll("[data-gallery-key]");
  for (const node of nodes) {
    if (node.classList.contains("removing")) continue;
    const before = beforePositions.get(node.dataset.galleryKey);
    if (!before) continue;
    const after = node.getBoundingClientRect();
    const deltaX = before.left - after.left;
    const deltaY = before.top - after.top;
    const scaleX = after.width > 0 ? before.width / after.width : 1;
    const scaleY = after.height > 0 ? before.height / after.height : 1;
    if (Math.abs(deltaX) < 0.5 && Math.abs(deltaY) < 0.5 && Math.abs(scaleX - 1) < 0.01 && Math.abs(scaleY - 1) < 0.01) {
      continue;
    }
    node.style.transition = "none";
    node.style.transformOrigin = "top left";
    node.style.transform = `translate(${deltaX}px, ${deltaY}px) scale(${scaleX}, ${scaleY})`;
    void node.offsetWidth;
    node.style.transition = "transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1)";
    node.style.transform = "";
    const cleanup = () => {
      node.style.transition = "";
      node.style.transformOrigin = "";
    };
    node.addEventListener("transitionend", cleanup, { once: true });
  }
}

function finalizeEnteringTile(tile) {
  requestAnimationFrame(() => {
    tile.classList.remove("entering");
  });
}

function cancelScheduledTileRemoval(tile) {
  const handle = tile?._galleryRemovalHandle;
  if (!handle) return;
  handle.cancelled = true;
  if (handle.timeoutId !== null) {
    window.clearTimeout(handle.timeoutId);
  }
  tile._galleryRemovalHandle = null;
  tile.classList.remove("removing");
}

function scheduleTileRemoval(tile) {
  if (!tile || tile.classList.contains("removing")) return;
  tile.classList.add("removing");
  const handle = { cancelled: false, timeoutId: null };
  tile._galleryRemovalHandle = handle;
  let done = false;
  const cleanup = () => {
    if (done) return;
    if (handle.cancelled) return;
    done = true;
    tile._galleryRemovalHandle = null;
    const before = captureGalleryPositions();
    if (tile.parentElement === galleryEl) {
      tile.remove();
    }
    scheduleGalleryRelayout({ animate: false });
    window.requestAnimationFrame(() => animateGalleryLayout(before));
  };
  tile.addEventListener("transitionend", cleanup, { once: true });
  handle.timeoutId = window.setTimeout(cleanup, 240);
}

function showZipProgressModal() {
  zipProgressModalEl.classList.remove("hidden");
  zipProgressModalEl.setAttribute("aria-hidden", "false");
  zipProgressCancelEl.focus();
}

function hideZipProgressModal() {
  zipProgressModalEl.classList.add("hidden");
  zipProgressModalEl.setAttribute("aria-hidden", "true");
}

function cancelZipDownload() {
  if (state.zipAbortController) {
    state.zipAbortController.abort();
  } else {
    hideZipProgressModal();
  }
}

function setGalleryColumns(count, options = {}) {
  const next = Math.max(3, Math.min(8, Number(count) || 4));
  const animate = Boolean(options.animate);
  state.galleryColumns = next;
  state.pendingGalleryColumns = next;
  galleryDensitySliderEl.value = String(next);
  document.documentElement.style.setProperty("--gallery-columns", String(next));
  try {
    window.localStorage.setItem(GALLERY_COLUMNS_STORAGE_KEY, String(next));
  } catch (_) {
  }
  scheduleGalleryRelayout({ animate });
}

function scheduleGalleryColumns(count) {
  state.pendingGalleryColumns = Math.max(3, Math.min(8, Number(count) || 4));
  galleryDensitySliderEl.value = String(state.pendingGalleryColumns);
  if (state.galleryColumnFrame !== null) {
    return;
  }
  state.galleryColumnFrame = window.requestAnimationFrame(() => {
    state.galleryColumnFrame = null;
    setGalleryColumns(state.pendingGalleryColumns, { animate: true });
  });
}

function getSelectedGalleryItems() {
  return state.galleryItems.filter((item) => state.selectedFilenames.has(item.filename));
}

function selectedImageCount() {
  return getSelectedGalleryItems().length;
}

function syncSelectedFilenames() {
  const available = new Set(state.galleryItems.map((item) => item.filename));
  state.selectedFilenames = new Set(
    [...state.selectedFilenames].filter((filename) => available.has(filename))
  );
}

function clearMultiSelection() {
  state.selectedFilenames.clear();
}

function updateMultiSelectControls() {
  const count = selectedImageCount();
  const active = state.multiSelectMode;
  multiSelectToggleEl.textContent = active ? "Cancel" : "Multiselection";
  multiSelectToggleEl.setAttribute("aria-pressed", String(active));
  multiSelectActionsEl.classList.toggle("hidden", !active);
  multiSelectCountEl.textContent = count === 1 ? "1 selected" : `${count} selected`;
  bulkDownloadButtonEl.disabled = count === 0;
  bulkDeleteButtonEl.disabled = count === 0;
  bulkUpscaleButtonEl.disabled = count === 0;
}

function toggleMultiSelectMode(force) {
  const next = typeof force === "boolean" ? force : !state.multiSelectMode;
  state.multiSelectMode = next;
  if (!next) {
    clearMultiSelection();
  }
  updateMultiSelectControls();
  renderGallery();
}

function toggleGallerySelection(filename) {
  const target = String(filename || "").trim();
  if (!target) return;
  if (state.selectedFilenames.has(target)) {
    state.selectedFilenames.delete(target);
  } else {
    state.selectedFilenames.add(target);
  }
  updateMultiSelectControls();
  renderGallery();
}

function resolveSourceFilename(item) {
  const direct = String(item?.source_filename || "").trim();
  if (direct) {
    return direct;
  }
  const sourceImage = String(item?.source_image || "").trim();
  if (!sourceImage) {
    return "";
  }
  const normalized = sourceImage.replaceAll("\\", "/");
  const pieces = normalized.split("/");
  return String(pieces[pieces.length - 1] || "").trim();
}

function isUpscaledItem(item) {
  if (!item) return false;
  const mode = String(item.mode || "").toLowerCase();
  return Boolean(resolveSourceFilename(item) || mode.includes("upscale"));
}

function canUpscaleItem(item) {
  return !isUpscaledItem(item);
}

function updateSettingsSummary() {
  const dimensions = parseResolution(resolutionSelectEl.value);
  const pieces = [
    `Resolution <span class="summary-value">${dimensions.width}x${dimensions.height}</span>`,
    `Enhancer <span class="summary-value">${state.promptEnhance ? "ON" : "OFF"}</span>`,
  ];

  if (state.freezeSeed && state.currentSeed !== null) {
    pieces.push(`Seed <span class="summary-value">${state.currentSeed}</span>`);
  }
  if (state.proceduralCreativity > 0) {
    pieces.push(
      `Creative Mode <span class="summary-value">${describeProceduralCreativity(state.proceduralCreativity)}</span>`,
    );
  }

  settingsSummaryEl.innerHTML = pieces
    .map((piece, index) => (index === 0 ? piece : `<span class="summary-sep">|</span> ${piece}`))
    .join(" ");
  updateTopbarOffset();
}

function updateReverseButton() {
  if (state.newestFirst) {
    reverseOrderButtonEl.textContent = "Newest First";
    reverseOrderButtonEl.classList.remove("reversed");
  } else {
    reverseOrderButtonEl.textContent = "Oldest First";
    reverseOrderButtonEl.classList.add("reversed");
  }
}

function updateColorSwatches() {
  const buttons = galleryColorFiltersEl.querySelectorAll("[data-color-filter]");
  for (const button of buttons) {
    const color = String(button.dataset.colorFilter || "").trim().toLowerCase();
    const active = color && color === state.activeColorFilter;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

async function setActiveColorFilter(color) {
  const normalized = GALLERY_COLOR_FILTERS.includes(String(color || "").trim().toLowerCase())
    ? String(color || "").trim().toLowerCase()
    : null;
  state.activeColorFilter = state.activeColorFilter === normalized ? null : normalized;
  updateColorSwatches();
  await loadImages();
}

function updateFreezeSeedButton() {
  if (state.freezeSeed) {
    freezeSeedButtonEl.textContent = `FREEZE SEED: ON (${state.currentSeed})`;
    freezeSeedButtonEl.classList.add("active");
  } else {
    freezeSeedButtonEl.textContent = "FREEZE SEED: OFF";
    freezeSeedButtonEl.classList.remove("active");
  }
}

function describeProceduralCreativity(level) {
  if (level <= 0) return "OFF";
  if (level === 1) return "Light";
  if (level === 2) return "Medium";
  return "Extreme";
}

function updateProceduralLatentControls() {
  proceduralLatentSliderEl.value = String(state.proceduralCreativity);
  proceduralLatentValueEl.textContent =
    `CREATIVE MODE: ${describeProceduralCreativity(state.proceduralCreativity)}`;
  proceduralLatentValueEl.classList.toggle("active", state.proceduralCreativity > 0);
  if (state.proceduralCreativity > 0) {
    proceduralLatentValueEl.style.color = "var(--lime)";
  } else {
    proceduralLatentValueEl.style.color = "#d8d8d8";
  }
}

function updatePromptEnhanceButton() {
  if (state.promptEnhance) {
    promptEnhanceButtonEl.textContent = "PROMPT ENHANCER: ON";
    promptEnhanceButtonEl.classList.add("active");
  } else {
    promptEnhanceButtonEl.textContent = "PROMPT ENHANCER: OFF";
    promptEnhanceButtonEl.classList.remove("active");
  }
}

function applyViewerTransform() {
  viewerImageEl.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  zoomLabelEl.textContent = `${state.zoom.toFixed(2)}x`;
}

function setZoom(value) {
  const clamped = Math.min(10, Math.max(0.01, value));
  state.zoom = clamped;
  if (state.zoom <= 1) {
    state.panX = 0;
    state.panY = 0;
  }
  applyViewerTransform();
}

function endViewerCompareHold() {
  if (!state.viewerCompareHolding) return;
  state.viewerCompareHolding = false;
  state.viewerCompareSourceFilename = null;
  const compareButton = viewerMetaEl.querySelector(".viewer-compare-hold");
  if (compareButton) {
    compareButton.classList.remove("active");
  }
  const item = getActiveViewerItem();
  if (!item) return;
  viewerImageEl.src = buildViewerImageUrl(item.filename);
  viewerImageEl.alt = item.prompt || "Generated image preview";
}

function beginViewerCompareHold() {
  if (viewerModalEl.classList.contains("hidden")) return;
  const item = getActiveViewerItem();
  if (!item) return;
  const sourceFilename = resolveSourceFilename(item);
  if (!sourceFilename) return;
  state.viewerCompareHolding = true;
  state.viewerCompareSourceFilename = sourceFilename;
  const compareButton = viewerMetaEl.querySelector(".viewer-compare-hold");
  if (compareButton) {
    compareButton.classList.add("active");
  }
  viewerImageEl.src = buildViewerImageUrl(sourceFilename);
  viewerImageEl.alt = `Original preview ${sourceFilename}`;
}

function hideViewer() {
  endViewerCompareHold();
  viewerModalEl.classList.add("hidden");
  viewerModalEl.setAttribute("aria-hidden", "true");
  state.viewerIndex = -1;
  state.viewerFilename = null;
  state.viewerPromptExpanded = false;
  state.dragging = false;
  viewerStageEl.classList.remove("dragging");
}

function updateViewerNavState() {
  const count = state.galleryItems.length;
  const hasPrev = state.viewerIndex > 0;
  const hasNext = state.viewerIndex >= 0 && state.viewerIndex < count - 1;
  viewerPrevButtonEl.disabled = !hasPrev;
  viewerNextButtonEl.disabled = !hasNext;
}

function applyViewerItemMeta(item) {
  const resolution = item.width && item.height ? `${item.width}x${item.height}` : "unknown";
  const upscaled = isUpscaledItem(item);
  viewerUpscaleButtonEl.classList.toggle("hidden", upscaled);
  viewerUpscaleButtonEl.disabled = upscaled;
  if (upscaled) {
    const sourceFilename = resolveSourceFilename(item);
    const compareAvailable = Boolean(sourceFilename);
    viewerMetaEl.classList.remove("expanded");
    viewerMetaEl.innerHTML = [
      `<span class="viewer-meta-source">Upscaled from ${escapeHtml(sourceFilename || "unknown image")}</span>`,
      '<span class="viewer-meta-sep">|</span>',
      `<button type="button" class="viewer-compare-hold" title="Hold to compare original"${
        compareAvailable ? "" : " disabled"
      }>${compareAvailable ? "HOLD TO SEE ORIGINAL" : "ORIGINAL NOT AVAILABLE"}</button>`,
      '<span class="viewer-meta-sep">|</span>',
      `<span>${escapeHtml(resolution)}</span>`,
    ].join(" ");
  } else {
    const timestamp = item.timestamp || item.created_at;
    const pack = item.model_pack || "n/a";
    const promptText = String(item.prompt || "").trim() || "(empty prompt)";
    const promptDisplay = state.viewerPromptExpanded ? promptText : shortPrompt(promptText, 140);
    const promptTitle = state.viewerPromptExpanded ? "Click to collapse prompt" : "Click to expand prompt";
    viewerMetaEl.classList.toggle("expanded", state.viewerPromptExpanded);
    viewerMetaEl.innerHTML = [
      `<span>${escapeHtml(formatTimestamp(timestamp))}</span>`,
      '<span class="viewer-meta-sep">|</span>',
      `<button type="button" class="viewer-meta-prompt${state.viewerPromptExpanded ? " expanded" : ""}" title="${promptTitle}">${escapeHtml(promptDisplay)}</button>`,
      '<span class="viewer-meta-sep">|</span>',
      `<span>${escapeHtml(resolution)}</span>`,
      '<span class="viewer-meta-sep">|</span>',
      `<span>${escapeHtml(pack)}</span>`,
    ].join(" ");
  }
  viewerDownloadEl.href = buildDownloadUrl(item.filename);
  viewerDownloadEl.setAttribute("download", item.filename);
}

function showViewer(item, index = -1) {
  if (index < 0) {
    index = state.galleryItems.findIndex((candidate) => candidate.filename === item.filename);
  }
  state.viewerIndex = index;
  state.viewerCompareHolding = false;
  state.viewerCompareSourceFilename = null;
  viewerImageEl.src = buildViewerImageUrl(item.filename);
  viewerImageEl.alt = item.prompt || "Generated image preview";
  state.viewerPromptExpanded = false;
  applyViewerItemMeta(item);
  state.panX = 0;
  state.panY = 0;
  state.viewerFilename = item.filename || null;
  setZoom(1.0);
  viewerModalEl.classList.remove("hidden");
  viewerModalEl.setAttribute("aria-hidden", "false");
  updateViewerNavState();
}

function getActiveViewerItem() {
  if (state.viewerIndex >= 0 && state.viewerIndex < state.galleryItems.length) {
    return state.galleryItems[state.viewerIndex];
  }
  if (state.viewerFilename) {
    return state.galleryItems.find((item) => item.filename === state.viewerFilename) || null;
  }
  return null;
}

function showViewerByOffset(direction) {
  if (viewerModalEl.classList.contains("hidden")) return;
  const nextIndex = state.viewerIndex + direction;
  if (nextIndex < 0 || nextIndex >= state.galleryItems.length) return;
  showViewer(state.galleryItems[nextIndex], nextIndex);
}

function syncViewerWithGallery() {
  if (viewerModalEl.classList.contains("hidden")) return;
  if (!state.viewerFilename) {
    hideViewer();
    return;
  }
  const index = state.galleryItems.findIndex((item) => item.filename === state.viewerFilename);
  if (index < 0) {
    hideViewer();
    return;
  }
  state.viewerIndex = index;
  endViewerCompareHold();
  applyViewerItemMeta(state.galleryItems[index]);
  updateViewerNavState();
}

async function copyTextToClipboard(text) {
  const value = String(text || "");
  if (!value) return false;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch (_) {
  }

  const helper = document.createElement("textarea");
  helper.value = value;
  helper.setAttribute("readonly", "readonly");
  helper.style.position = "fixed";
  helper.style.left = "-9999px";
  document.body.append(helper);
  helper.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch (_) {
    copied = false;
  }
  helper.remove();
  return copied;
}

function onViewerUsePrompt() {
  const item = getActiveViewerItem();
  if (!item) return;
  promptInputEl.value = String(item.prompt || "");
  updateTopbarOffset();
  hideViewer();
  promptInputEl.focus();
  setStatus("Loaded prompt into top bar.");
}

async function onViewerCopyPrompt() {
  const item = getActiveViewerItem();
  if (!item) return;
  const copied = await copyTextToClipboard(item.prompt || "");
  if (copied) {
    setStatus("Prompt copied to clipboard.");
  } else {
    setStatus("Failed to copy prompt to clipboard.", true);
  }
}

function onViewerUpscale() {
  const item = getActiveViewerItem();
  if (!item) return;
  enqueueUpscaleFromItem(item);
}

function showConfirmModal(message, onConfirm, confirmLabel = "Confirm", cancelLabel = "Cancel") {
  state.confirmAction = onConfirm;
  confirmModalMessageEl.textContent = message;
  confirmModalConfirmEl.textContent = confirmLabel;
  confirmModalCancelEl.textContent = cancelLabel;
  confirmModalEl.classList.remove("hidden");
  confirmModalEl.setAttribute("aria-hidden", "false");
  confirmModalConfirmEl.focus();
}

function hideConfirmModal() {
  state.confirmAction = null;
  confirmModalEl.classList.add("hidden");
  confirmModalEl.setAttribute("aria-hidden", "true");
}

function removeQueuedJob(placeholderId) {
  const target = String(placeholderId || "").trim();
  if (!target) return false;
  const previousLength = state.queue.length;
  state.queue = state.queue.filter((job) => job.placeholderId !== target);
  if (state.queue.length === previousLength) {
    return false;
  }
  persistClientQueueState();
  renderGallery();
  updateGenerateButtonState();
  return true;
}

async function requestActiveJobCancel(job) {
  if (!job || !job.placeholderId || !state.activeJob || state.activeJob.placeholderId !== job.placeholderId) {
    return false;
  }
  if (state.activeJob.status !== "cancelling") {
    state.activeJob.status = "cancelling";
    persistClientQueueState();
    renderGallery();
  }
  const response = await apiFetch("/client-jobs/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: job.placeholderId }),
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(formatApiError(payload, "Failed to cancel active job."));
  }
  setStatus(String(payload?.message || "Cancellation requested."));
  return true;
}

async function cancelPendingJobById(placeholderId) {
  const target = String(placeholderId || "").trim();
  if (!target) return false;
  if (state.activeJob && state.activeJob.placeholderId === target) {
    return requestActiveJobCancel(state.activeJob);
  }
  const removed = removeQueuedJob(target);
  if (removed) {
    setStatus("Queued job cancelled.");
  }
  return removed;
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

function pendingJobLabel(job, queuePosition) {
  const isUpscale = job.kind === "upscale";
  if (job.status === "cancelling") {
    return "CANCELLING...";
  }
  if (job.status === "generating") {
    return isUpscale ? "UPSCALING..." : "GENERATING...";
  }
  if (job.status === "upscaling") {
    return "UPSCALING...";
  }
  if (queuePosition >= 0) {
    const prefix = isUpscale ? "UPSCALE QUEUED" : "QUEUED";
    return `${prefix} (${queuePosition + 1})`;
  }
  return isUpscale ? "UPSCALE QUEUED..." : "QUEUED...";
}

function createPendingTile(job) {
  const tile = document.createElement("article");
  tile.className = "tile pending";
  tile.dataset.galleryKey = galleryKeyForPending(job);
  tile.dataset.placeholderId = job.placeholderId;

  const canvas = document.createElement("div");
  canvas.className = "tile-placeholder";
  tile.append(canvas);

  const spinner = document.createElement("div");
  spinner.className = "tile-spinner";
  canvas.append(spinner);

  const cancelButton = document.createElement("button");
  cancelButton.className = "tile-pending-cancel";
  cancelButton.type = "button";
  cancelButton.addEventListener("click", (event) => {
    event.stopPropagation();
    const placeholderId = tile.dataset.placeholderId || "";
    cancelPendingJobById(placeholderId).catch((error) => setStatus(String(error?.message || error), true));
  });
  canvas.append(cancelButton);
  updatePendingTile(tile, job);
  tile.classList.add("entering");
  finalizeEnteringTile(tile);
  return tile;
}

function updatePendingTile(tile, job) {
  cancelScheduledTileRemoval(tile);
  tile.dataset.galleryKey = galleryKeyForPending(job);
  tile.dataset.placeholderId = job.placeholderId;
  tile.dataset.aspectWidth = String(Math.max(1, Number(job.width) || 1));
  tile.dataset.aspectHeight = String(Math.max(1, Number(job.height) || 1));
  tile.classList.add("pending");
  tile.classList.toggle("generating", isActiveJobStatus(job.status));
  tile.classList.toggle("queued", !isActiveJobStatus(job.status));
  tile.classList.toggle("cancelling", job.status === "cancelling");
  const canvas = tile.querySelector(".tile-placeholder");
  const spinner = tile.querySelector(".tile-spinner");
  const cancelButton = tile.querySelector(".tile-pending-cancel");
  if (canvas) {
    canvas.style.aspectRatio = `${job.width} / ${job.height}`;
  }
  if (spinner) {
    const queuePosition = state.queue.findIndex((queued) => queued.placeholderId === job.placeholderId);
    spinner.textContent = pendingJobLabel(job, queuePosition);
  }
  if (cancelButton instanceof HTMLButtonElement) {
    cancelButton.textContent = job.status === "cancelling" ? "Cancelling" : "Cancel";
    cancelButton.disabled = job.status === "cancelling";
    cancelButton.setAttribute("aria-label", `Cancel ${job.kind === "upscale" ? "upscale" : "generation"} job`);
    cancelButton.title = "Cancel";
  }
}

function dropMissingGalleryItem(filename) {
  const target = String(filename || "").trim();
  if (!target) return;
  const previousCount = state.galleryItems.length;
  state.galleryItems = state.galleryItems.filter((item) => item.filename !== target);
  if (state.viewerFilename === target) {
    hideViewer();
  }
  if (state.galleryItems.length !== previousCount) {
    renderGallery();
    syncViewerWithGallery();
  }
}

function createImageTile(item) {
  const tile = document.createElement("article");
  tile.className = "tile";

  const image = document.createElement("img");
  image.loading = "lazy";
  image.addEventListener("error", () => {
    tile.remove();
    dropMissingGalleryItem(tile.dataset.filename || "");
  });

  const overlay = document.createElement("div");
  overlay.className = "tile-overlay";
  const badges = document.createElement("div");
  badges.className = "tile-badges";
  const upscaleBadge = document.createElement("span");
  upscaleBadge.className = "tile-badge tile-badge-upscaled";
  upscaleBadge.title = "Upscaled";
  upscaleBadge.setAttribute("aria-hidden", "true");
  upscaleBadge.innerHTML =
    '<svg viewBox="0 0 24 24" focusable="false"><path d="M4 9V4h5v2H6v3H4zm10-5h6v6h-2V6h-4V4zM4 14h2v4h4v2H4v-6zm14 4v-4h2v6h-6v-2h4z"/></svg>';
  badges.append(upscaleBadge);

  const selectBadge = document.createElement("span");
  selectBadge.className = "tile-badge tile-select-badge";
  selectBadge.setAttribute("aria-hidden", "true");
  badges.append(selectBadge);

  const meta = document.createElement("div");
  meta.className = "tile-meta";

  const actions = document.createElement("div");
  actions.className = "tile-actions";
  const primaryActions = document.createElement("div");
  primaryActions.className = "tile-primary-actions";

  const download = document.createElement("a");
  download.className = "tile-download";
  download.textContent = "Download";
  download.addEventListener("click", (event) => event.stopPropagation());

  const del = document.createElement("button");
  del.className = "tile-delete";
  del.type = "button";
  del.title = "Delete image";
  del.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l1 2h5v2H3V5h5l1-2zm1 7h2v8h-2v-8zm4 0h2v8h-2v-8zM7 10h2v8H7v-8z"/></svg>';
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    const filename = tile.dataset.filename || "";
    showConfirmModal(`Delete "${filename}"? This cannot be undone.`, async () => {
      await deleteImage(filename);
    }, "Delete");
  });

  primaryActions.append(download);
  const upscale = document.createElement("button");
  upscale.className = "tile-upscale";
  upscale.type = "button";
  upscale.textContent = "Upscale";
  upscale.addEventListener("click", (event) => {
    event.stopPropagation();
    const liveItem = getGalleryImageItem(tile.dataset.filename);
    if (liveItem) {
      enqueueUpscaleFromItem(liveItem);
    }
  });
  primaryActions.append(upscale);
  actions.append(primaryActions, del);
  overlay.append(meta, actions);
  tile.append(image, badges, overlay);
  tile.addEventListener("click", () => {
    const filename = tile.dataset.filename || "";
    if (state.multiSelectMode) {
      toggleGallerySelection(filename);
      return;
    }
    const liveItem = getGalleryImageItem(filename);
    if (!liveItem) return;
    const liveIndex = state.galleryItems.findIndex((candidate) => candidate.filename === filename);
    showViewer(liveItem, liveIndex);
  });
  updateImageTile(tile, item);
  tile.classList.add("entering");
  finalizeEnteringTile(tile);
  return tile;
}

function updateImageTile(tile, item) {
  cancelScheduledTileRemoval(tile);
  const upscaled = isUpscaledItem(item);
  const selected = state.selectedFilenames.has(item.filename);
  const aspectWidth = Math.max(1, Number(item.width) || 1);
  const aspectHeight = Math.max(1, Number(item.height) || 1);
  tile.dataset.galleryKey = galleryKeyForImage(item);
  tile.dataset.filename = item.filename;
  tile.dataset.aspectWidth = String(aspectWidth);
  tile.dataset.aspectHeight = String(aspectHeight);
  tile.classList.toggle("multiselect-active", state.multiSelectMode);
  tile.classList.toggle("selected", selected);

  const image = tile.querySelector("img");
  if (image) {
    const nextSrc = buildImageUrl(item.filename);
    if (image.src !== new URL(nextSrc, window.location.origin).toString()) {
      image.src = nextSrc;
    }
    image.alt = item.prompt || "Generated image";
  }

  const timestamp = item.timestamp || item.created_at;
  const resolution = item.width && item.height ? `${item.width}x${item.height}` : "unknown";
  const meta = tile.querySelector(".tile-meta");
  if (meta) {
    meta.textContent = `${formatGalleryTimestamp(timestamp)} | ${shortPrompt(item.prompt, 60)} | ${resolution}`;
  }

  const upscaleBadge = tile.querySelector(".tile-badge-upscaled");
  if (upscaleBadge) {
    upscaleBadge.style.display = upscaled ? "" : "none";
  }

  const selectBadge = tile.querySelector(".tile-select-badge");
  if (selectBadge) {
    selectBadge.textContent = selected ? "✓" : "";
    selectBadge.style.display = state.multiSelectMode ? "" : "none";
  }

  const download = tile.querySelector(".tile-download");
  if (download) {
    download.href = buildDownloadUrl(item.filename);
    download.setAttribute("download", item.filename);
  }

  const del = tile.querySelector(".tile-delete");
  if (del) {
    del.setAttribute("aria-label", `Delete ${item.filename}`);
  }

  const upscale = tile.querySelector(".tile-upscale");
  if (upscale) {
    upscale.style.display = canUpscaleItem(item) ? "" : "none";
  }
}

function buildDesiredGalleryEntries() {
  const pendingJobs = [...state.pendingJobs];
  const activePending = pendingJobs.find((job) => isActiveJobStatus(job.status)) || null;
  const queuedPending = pendingJobs
    .filter((job) => !isActiveJobStatus(job.status))
    .sort((left, right) =>
      state.newestFirst ? right.enqueuedAt - left.enqueuedAt : left.enqueuedAt - right.enqueuedAt
    );
  const pendingDisplay = state.newestFirst
    ? [...(activePending ? [activePending] : []), ...queuedPending]
    : [...queuedPending, ...(activePending ? [activePending] : [])];
  const pending = state.newestFirst ? pendingDisplay : [];
  const items = [...state.galleryItems];
  const trailingPending = state.newestFirst ? [] : pendingDisplay;
  const desired = [];
  for (const job of pending) {
    desired.push({ key: galleryKeyForPending(job), kind: "pending", value: job });
  }
  for (const item of items) {
    desired.push({ key: galleryKeyForImage(item), kind: "image", value: item });
  }
  for (const job of trailingPending) {
    desired.push({ key: galleryKeyForPending(job), kind: "pending", value: job });
  }
  return desired;
}

function renderGallery() {
  syncSelectedFilenames();
  updateMultiSelectControls();
  updateGalleryCount(state.galleryItems.length);
  const desiredEntries = buildDesiredGalleryEntries();
  const hasContent = desiredEntries.length > 0;
  emptyStateEl.classList.toggle("hidden", hasContent);
  if (!hasContent) {
    galleryEl.style.height = "";
    for (const tile of galleryEl.querySelectorAll("[data-gallery-key]")) {
      scheduleTileRemoval(tile);
    }
    return;
  }

  const existingMap = getGalleryNodeMap();
  for (const entry of desiredEntries) {
    let tile = existingMap.get(entry.key) || null;
    if (!tile) {
      tile = entry.kind === "pending" ? createPendingTile(entry.value) : createImageTile(entry.value);
    } else if (entry.kind === "pending") {
      updatePendingTile(tile, entry.value);
    } else {
      updateImageTile(tile, entry.value);
    }
    galleryEl.append(tile);
    existingMap.delete(entry.key);
  }

  for (const tile of existingMap.values()) {
    scheduleTileRemoval(tile);
  }

  scheduleGalleryRelayout({ animate: true });
}

function updateGalleryCount(imageCount = state.galleryItems.length) {
  const count = Math.max(0, Number(imageCount) || 0);
  galleryCountEl.textContent = count === 1 ? "1 image" : `${count} images`;
}

function toTimestamp(item) {
  const candidate = item.timestamp || item.created_at || "";
  const parsed = Date.parse(candidate);
  if (!Number.isFinite(parsed)) return 0;
  return parsed;
}

function sortItems(items) {
  const sorted = [...items];
  sorted.sort((a, b) => {
    const aTime = toTimestamp(a);
    const bTime = toTimestamp(b);
    if (aTime !== bTime) {
      return state.newestFirst ? bTime - aTime : aTime - bTime;
    }
    const aId = Number(a.id || 0);
    const bId = Number(b.id || 0);
    return state.newestFirst ? bId - aId : aId - bId;
  });
  return sorted;
}

function resolveSeedForGeneration() {
  if (state.freezeSeed) {
    if (state.currentSeed === null) {
      state.currentSeed = randomSeed();
      updateFreezeSeedButton();
      updateSettingsSummary();
    }
    return state.currentSeed;
  }
  state.currentSeed = randomSeed();
  return state.currentSeed;
}

function totalOutstandingJobs() {
  return state.queue.length + (state.activeJob ? 1 : 0);
}

function updateGenerateButtonState() {
  const outstanding = totalOutstandingJobs();
  const queueFull = outstanding >= state.maxQueuedGenerations;
  generateButtonEl.disabled = queueFull;
  let label = "GENERATE";
  if (queueFull) {
    label = "QUEUE FULL";
  } else if (outstanding > 0) {
    label = `GENERATE (${outstanding}/${state.maxQueuedGenerations})`;
  }
  generateButtonLabelEl.textContent = label;
  generateButtonEl.setAttribute("aria-label", label);
  generateButtonEl.title = label;
}

async function loadImages() {
  const requestSeq = ++state.galleryLoadRequestSeq;
  const query = new URLSearchParams();
  query.set("limit", "500");
  query.set("newest_first", "true");

  const filterValue = String(filterInputEl.value || "").trim();
  if (filterValue) {
    query.set("prompt", filterValue);
  }
  if (state.activeColorFilter) {
    query.set("color", state.activeColorFilter);
  }

  const response = await apiFetch(`/images?${query.toString()}`, { cache: "no-store" });
  if (!response.ok) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    throw new Error(formatApiError(payload, "Failed to load gallery."));
  }
  const payload = await response.json();
  if (requestSeq !== state.galleryLoadRequestSeq) {
    return;
  }
  state.galleryColorCacheActive = Boolean(payload?.color_cache?.active);
  scheduleGalleryColorCachePoll();
  state.galleryItems = sortItems(payload.items || []);
  syncSelectedFilenames();
  renderGallery();
  syncViewerWithGallery();
  syncGalleryColorCacheStatusLine();
}

async function deleteImage(filename, options = {}) {
  const skipReload = Boolean(options.skipReload);
  const suppressStatus = Boolean(options.suppressStatus);
  const existingItems = state.galleryItems;
  const hadItem = existingItems.some((item) => item.filename === filename);
  if (hadItem) {
    state.selectedFilenames.delete(filename);
    state.galleryItems = existingItems.filter((item) => item.filename !== filename);
    if (state.viewerFilename === filename) {
      hideViewer();
    }
    renderGallery();
    syncViewerWithGallery();
  }

  let payload = null;
  const response = await apiFetch(`/images/${encodeURIComponent(filename)}?confirm=DELETE`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: "DELETE" }),
  });
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    if (hadItem) {
      state.galleryItems = existingItems;
      renderGallery();
      syncViewerWithGallery();
    }
    throw new Error(formatApiError(payload, "Image deletion failed."));
  }
  if (!skipReload) {
    await loadImages();
  }
  if (!suppressStatus) {
    setStatus(`Deleted ${filename}.`);
  }
}

async function deleteSelectedImages() {
  const selectedItems = getSelectedGalleryItems();
  const count = selectedItems.length;
  if (count === 0) {
    setStatus("No images selected.", true);
    return;
  }
  for (const item of selectedItems) {
    await deleteImage(item.filename, { skipReload: true, suppressStatus: true });
  }
  await loadImages();
  clearMultiSelection();
  updateMultiSelectControls();
  renderGallery();
  setStatus(`Deleted ${count} image${count === 1 ? "" : "s"}.`);
}

function queueSelectedUpscales() {
  const selectedItems = getSelectedGalleryItems();
  if (selectedItems.length === 0) {
    setStatus("Select at least one image to upscale.", true);
    return false;
  }
  if (selectedItems.length > 5) {
    setStatus("Bulk upscale is limited to 5 selected images.", true);
    return false;
  }
  const invalidItems = selectedItems.filter((item) => !canUpscaleItem(item));
  if (invalidItems.length > 0) {
    setStatus("Bulk upscale only works on original images that have not already been upscaled.", true);
    return false;
  }
  if (totalOutstandingJobs() + selectedItems.length > state.maxQueuedGenerations) {
    setStatus(`Bulk upscale would exceed the queue limit of ${state.maxQueuedGenerations}.`, true);
    return false;
  }
  let queued = 0;
  for (const item of selectedItems) {
    if (enqueueUpscaleFromItem(item)) {
      queued += 1;
    }
  }
  if (queued > 0) {
    clearMultiSelection();
    updateMultiSelectControls();
    renderGallery();
    setStatus(`Queued ${queued} upscale job${queued === 1 ? "" : "s"}.`);
    return true;
  }
  return false;
}

async function downloadSelectedImagesZip() {
  const selectedItems = getSelectedGalleryItems();
  if (selectedItems.length === 0) {
    setStatus("No images selected.", true);
    return;
  }
  const controller = new AbortController();
  state.zipAbortController = controller;
  showZipProgressModal();
  try {
    const response = await apiFetch("/images/download-zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filenames: selectedItems.map((item) => item.filename) }),
      signal: controller.signal,
    });
    if (!response.ok) {
      let payload = null;
      try {
        payload = await response.json();
      } catch (_) {
        payload = null;
      }
      throw new Error(formatApiError(payload, "ZIP download failed."));
    }
    const blob = await response.blob();
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `${state.clientId}_selection.zip`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1500);
    setStatus(`Prepared ZIP download for ${selectedItems.length} image${selectedItems.length === 1 ? "" : "s"}.`);
  } catch (error) {
    if (error && error.name === "AbortError") {
      setStatus("ZIP download cancelled.");
      return;
    }
    throw error;
  } finally {
    if (state.zipAbortController === controller) {
      state.zipAbortController = null;
    }
    hideZipProgressModal();
  }
}

function enqueueGenerationFromPrompt() {
  const prompt = String(promptInputEl.value || "").trim();
  if (!prompt) {
    setStatus("Prompt is required.", true);
    return false;
  }

  if (totalOutstandingJobs() >= state.maxQueuedGenerations) {
    updateGenerateButtonState();
    setStatus(`Queue is full (${state.maxQueuedGenerations}/${state.maxQueuedGenerations}).`, true);
    return false;
  }

  const dimensions = parseResolution(resolutionSelectEl.value);
  const seed = resolveSeedForGeneration();
  const placeholderId = `pending_${Date.now()}_${Math.random().toString(16).slice(2)}`;

  const job = {
    kind: "generate",
    placeholderId,
    prompt,
    width: dimensions.width,
    height: dimensions.height,
    seed,
    enhance_prompt: state.promptEnhance,
    procedural_creativity: state.proceduralCreativity,
    enqueuedAt: Date.now(),
    remoteInFlight: false,
  };

  state.queue.push(job);
  persistClientQueueState();
  renderGallery();
  updateGenerateButtonState();
  const outstanding = totalOutstandingJobs();
  setStatus(`Queued ${dimensions.width}x${dimensions.height} (seed ${seed}). Queue ${outstanding}/${state.maxQueuedGenerations}.`);
  processGenerationQueue().catch((error) => setStatus(String(error?.message || error), true));
  return true;
}

function enqueueUpscaleFromItem(item) {
  if (!canUpscaleItem(item)) {
    setStatus("Upscale blocked: source image is already upscaled.", true);
    return false;
  }
  const sourceFilename = String(item?.filename || "").trim();
  if (!sourceFilename) {
    setStatus("Upscale failed: invalid source image.", true);
    return false;
  }

  if (totalOutstandingJobs() >= state.maxQueuedGenerations) {
    updateGenerateButtonState();
    setStatus(`Queue is full (${state.maxQueuedGenerations}/${state.maxQueuedGenerations}).`, true);
    return false;
  }

  const sourceWidth = Number(item.width) || 1024;
  const sourceHeight = Number(item.height) || 1024;
  const targetWidth = Math.max(64, sourceWidth * 2);
  const targetHeight = Math.max(64, sourceHeight * 2);
  const seed = resolveSeedForGeneration();
  const placeholderId = `pending_upscale_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  const preferredPack = String(item.model_pack || item.pack || "").trim() || null;

  const job = {
    kind: "upscale",
    placeholderId,
    filename: sourceFilename,
    width: targetWidth,
    height: targetHeight,
    seed,
    pack: preferredPack,
    enhance_prompt: state.promptEnhance,
    source_filename: sourceFilename,
    enqueuedAt: Date.now(),
    remoteInFlight: false,
  };

  state.queue.push(job);
  persistClientQueueState();
  renderGallery();
  updateGenerateButtonState();
  const outstanding = totalOutstandingJobs();
  setStatus(`Queued upscale for ${sourceFilename} -> ${targetWidth}x${targetHeight}. Queue ${outstanding}/${state.maxQueuedGenerations}.`);
  processGenerationQueue().catch((error) => setStatus(String(error?.message || error), true));
  return true;
}

async function processGenerationQueue() {
  if (state.queueWorkerRunning) return;
  state.queueWorkerRunning = true;

  try {
    while (state.queue.length > 0 || state.activeJob) {
      if (state.activeJob && state.activeJob.remoteInFlight) {
        scheduleClientJobPoll();
        break;
      }
      if (!state.activeJob) {
        state.activeJob = state.queue.shift() || null;
        if (!state.activeJob) break;
        state.activeJob.status = state.activeJob.kind === "upscale" ? "upscaling" : "generating";
        state.activeJob.remoteInFlight = false;
        persistClientQueueState();
        renderGallery();
        updateGenerateButtonState();
      }

      const job = state.activeJob;
      if (!job) continue;

      try {
        const isUpscaleJob = job.kind === "upscale";
        if (isUpscaleJob) {
          setStatus(`Upscaling ${job.filename} -> ${job.width}x${job.height} (seed ${job.seed})...`);
        } else {
          setStatus(`Generating ${job.width}x${job.height} (seed ${job.seed})...`);
        }
        const endpoint = isUpscaleJob ? "/upscale" : "/generate";
        const payloadBody = isUpscaleJob
          ? {
              job_id: job.placeholderId,
              filename: job.filename,
              pack: job.pack,
              seed: job.seed,
              enhance_prompt: job.enhance_prompt,
            }
          : {
              job_id: job.placeholderId,
              prompt: job.prompt,
              width: job.width,
              height: job.height,
              seed: job.seed,
              enhance_prompt: job.enhance_prompt,
              procedural_creativity: Number(job.procedural_creativity || 0),
            };
        const response = await apiFetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payloadBody),
        });
        const payload = await response.json();
        if (!response.ok) {
          if (response.status === 409) {
            setStatus(isUpscaleJob ? "Upscale cancelled." : "Generation cancelled.");
            state.activeJob = null;
            persistClientQueueState();
            await loadImages();
            continue;
          }
          throw new Error(formatApiError(payload, isUpscaleJob ? "Upscale failed." : "Generation failed."));
        }
        if (isUpscaleJob) {
          const source = String(payload.source_filename || job.filename || "source image");
          setStatus(`Upscaled ${source} -> ${payload.filename} in ${payload.duration_ms} ms (seed ${payload.seed}).`);
        } else if (payload.prompt_enhanced) {
          setStatus(`Prompt enhanced, saved ${payload.filename} in ${payload.duration_ms} ms (seed ${payload.seed}).`);
        } else {
          setStatus(`Saved ${payload.filename} in ${payload.duration_ms} ms (seed ${payload.seed}).`);
        }
        state.activeJob = null;
        persistClientQueueState();
        await loadImages();
      } catch (error) {
        state.activeJob = null;
        persistClientQueueState();
        renderGallery();
        setStatus(String(error?.message || error), true);
      } finally {
        updateGenerateButtonState();
      }
    }
  } finally {
    state.queueWorkerRunning = false;
    updateGenerateButtonState();
  }
}

async function onDeleteGallery() {
  const confirmation = window.prompt("Type DELETE to confirm full gallery deletion:");
  if (confirmation === null) return;

  try {
    const encoded = encodeURIComponent(confirmation);
    const response = await apiFetch(`/gallery?confirm=${encoded}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: confirmation }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(payload, "Gallery deletion failed."));
    }
    hideViewer();
    state.galleryItems = [];
    state.queue = [];
    state.activeJob = null;
    state.pendingJobs = [];
    persistClientQueueState();
    clearMultiSelection();
    renderGallery();
    emptyStateEl.classList.remove("hidden");
    await loadImages();
    setStatus(
      `Deleted ${payload.deleted_files} file(s), removed ${payload.deleted_rows} index row(s), remaining ${payload.remaining_rows}.`
    );
  } catch (error) {
    setStatus(String(error?.message || error), true);
  }
}

async function onKillServer() {
  let payload = null;
  startDisconnectEffect();
  try {
    const response = await apiFetch("/server/kill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      stopDisconnectEffect();
      throw new Error(formatApiError(payload, "Server shutdown failed."));
    }
  } catch (error) {
    if (error instanceof TypeError) {
      return;
    }
    stopDisconnectEffect();
    setStatus(String(error?.message || error), true);
  }
}

function onFilterChanged() {
  if (state.filterTimer) {
    clearTimeout(state.filterTimer);
  }
  state.filterTimer = setTimeout(() => {
    loadImages().catch((error) => setStatus(String(error?.message || error), true));
  }, 220);
}

function applyOrientationButtonState() {
  const toggles = orientationToggleEl.querySelectorAll(".toggle-option");
  for (const button of toggles) {
    const orientation = button.dataset.orientation || "portrait";
    button.classList.toggle("active", orientation === state.orientation);
  }
}

function toggleFreezeSeed() {
  state.freezeSeed = !state.freezeSeed;
  if (state.freezeSeed && state.currentSeed === null) {
    state.currentSeed = randomSeed();
  }
  updateFreezeSeedButton();
  updateSettingsSummary();
}

function setProceduralCreativity(level) {
  const next = Math.max(0, Math.min(3, Number(level) || 0));
  state.proceduralCreativity = next;
  updateProceduralLatentControls();
  updateSettingsSummary();
}

function togglePromptEnhance() {
  state.promptEnhance = !state.promptEnhance;
  updatePromptEnhanceButton();
  updateSettingsSummary();
}

function beginDrag(event) {
  if (viewerModalEl.classList.contains("hidden")) return;
  if (event.button !== 0) return;
  state.dragging = true;
  state.dragStartX = event.clientX;
  state.dragStartY = event.clientY;
  state.dragBaseX = state.panX;
  state.dragBaseY = state.panY;
  viewerStageEl.classList.add("dragging");
  if (typeof viewerStageEl.setPointerCapture === "function") {
    try {
      viewerStageEl.setPointerCapture(event.pointerId);
    } catch (_) {
    }
  }
}

function moveDrag(event) {
  if (!state.dragging) return;
  if (typeof event.buttons === "number" && event.buttons === 0) {
    endDrag();
    return;
  }
  const dx = event.clientX - state.dragStartX;
  const dy = event.clientY - state.dragStartY;
  state.panX = state.dragBaseX + dx;
  state.panY = state.dragBaseY + dy;
  applyViewerTransform();
}

function endDrag(event) {
  if (!state.dragging) return;
  state.dragging = false;
  viewerStageEl.classList.remove("dragging");
  if (event && typeof viewerStageEl.releasePointerCapture === "function") {
    try {
      viewerStageEl.releasePointerCapture(event.pointerId);
    } catch (_) {
    }
  }
}

async function bootstrap() {
  try {
    restoreClientQueueState();
    setGalleryColumns(state.galleryColumns);
    updateTopbarOffset();
    updateReverseButton();
    updateColorSwatches();
    applyOrientationButtonState();
    updateFreezeSeedButton();
    updateProceduralLatentControls();
    updatePromptEnhanceButton();
    updateMultiSelectControls();
    updateSettingsSummary();
    updateViewerNavState();
    updateGenerateButtonState();
    await loadImages();
    await refreshClientJobState();
    if (state.queue.length > 0 || state.activeJob) {
      renderGallery();
      processGenerationQueue().catch((error) => setStatus(String(error?.message || error), true));
    } else if (state.galleryColorCacheActive) {
      setStatus(GALLERY_COLOR_CACHE_STATUS_MESSAGE);
    } else {
      setStatus("Ready.");
    }
  } catch (error) {
    setStatus(String(error?.message || error), true);
  }
}

settingsButtonEl.addEventListener("click", toggleSettingsVisible);
generateButtonEl.addEventListener("click", () => {
  enqueueGenerationFromPrompt();
});
freezeSeedButtonEl.addEventListener("click", toggleFreezeSeed);
proceduralLatentSliderEl.addEventListener("input", () => {
  setProceduralCreativity(proceduralLatentSliderEl.value);
});
promptEnhanceButtonEl.addEventListener("click", togglePromptEnhance);

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (!isSettingsOpen()) return;
  if (settingsPanelEl.contains(target) || settingsButtonEl.contains(target)) return;
  setSettingsVisible(false);
});

orientationToggleEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement)) return;
  const orientation = target.dataset.orientation;
  if (orientation !== "portrait" && orientation !== "landscape") return;
  state.orientation = orientation;
  applyOrientationButtonState();
  updateSettingsSummary();
});

resolutionSelectEl.addEventListener("change", () => {
  updateSettingsSummary();
});

promptInputEl.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  enqueueGenerationFromPrompt();
});
promptInputEl.addEventListener("input", updateTopbarOffset);
promptInputEl.addEventListener("mouseup", updateTopbarOffset);
promptInputEl.addEventListener("touchend", updateTopbarOffset);
window.addEventListener("resize", () => {
  updateTopbarOffset();
  scheduleGalleryRelayout({ animate: true });
});
window.addEventListener(
  "scroll",
  () => {
    if (isSettingsOpen()) {
      positionSettingsPanel();
    }
  },
  true
);
if (window.ResizeObserver) {
  const observer = new ResizeObserver(() => updateTopbarOffset());
  observer.observe(topbarEl);
}

filterInputEl.addEventListener("input", onFilterChanged);
reverseOrderButtonEl.addEventListener("click", () => {
  state.newestFirst = !state.newestFirst;
  updateReverseButton();
  loadImages().catch((error) => setStatus(String(error?.message || error), true));
});
galleryColorFiltersEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const button = target.closest("[data-color-filter]");
  if (!(button instanceof HTMLButtonElement)) return;
  setActiveColorFilter(button.dataset.colorFilter || null).catch((error) =>
    setStatus(String(error?.message || error), true)
  );
});
galleryDensitySliderEl.addEventListener("input", () => {
  scheduleGalleryColumns(galleryDensitySliderEl.value);
});
multiSelectToggleEl.addEventListener("click", () => {
  toggleMultiSelectMode();
});
bulkUpscaleButtonEl.addEventListener("click", () => {
  queueSelectedUpscales();
});
bulkDownloadButtonEl.addEventListener("click", () => {
  downloadSelectedImagesZip().catch((error) => setStatus(String(error?.message || error), true));
});
bulkDeleteButtonEl.addEventListener("click", () => {
  const count = selectedImageCount();
  if (count === 0) {
    setStatus("No images selected.", true);
    return;
  }
  showConfirmModal(
    `Delete ${count} selected image${count === 1 ? "" : "s"}? This cannot be undone.`,
    async () => {
      await deleteSelectedImages();
    },
    "Delete"
  );
});
deleteGalleryButtonEl.addEventListener("click", onDeleteGallery);
killServerButtonEl.addEventListener("click", () => {
  showConfirmModal("Kill the server now? This will disconnect the web UI.", onKillServer, "Kill Server");
});

viewerCloseButtonEl.addEventListener("click", hideViewer);
viewerUsePromptButtonEl.addEventListener("click", onViewerUsePrompt);
viewerCopyPromptButtonEl.addEventListener("click", () => {
  onViewerCopyPrompt().catch((error) => setStatus(String(error?.message || error), true));
});
viewerUpscaleButtonEl.addEventListener("click", onViewerUpscale);
viewerPrevButtonEl.addEventListener("pointerdown", (event) => event.stopPropagation());
viewerNextButtonEl.addEventListener("pointerdown", (event) => event.stopPropagation());
viewerPrevButtonEl.addEventListener("click", (event) => {
  event.stopPropagation();
  showViewerByOffset(-1);
});
viewerNextButtonEl.addEventListener("click", (event) => {
  event.stopPropagation();
  showViewerByOffset(1);
});
viewerMetaEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (!target.closest(".viewer-meta-prompt")) return;
  const item = getActiveViewerItem();
  if (!item) return;
  state.viewerPromptExpanded = !state.viewerPromptExpanded;
  applyViewerItemMeta(item);
});
viewerMetaEl.addEventListener("pointerdown", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (!target.closest(".viewer-compare-hold")) return;
  event.preventDefault();
  beginViewerCompareHold();
});
viewerMetaEl.addEventListener("pointerup", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (!target.closest(".viewer-compare-hold")) return;
  event.preventDefault();
  endViewerCompareHold();
});
viewerMetaEl.addEventListener("pointercancel", () => {
  endViewerCompareHold();
});
viewerMetaEl.addEventListener("pointerleave", () => {
  endViewerCompareHold();
});
viewerMetaEl.addEventListener("keydown", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (!target.closest(".viewer-compare-hold")) return;
  if (event.key !== " " && event.key !== "Enter") return;
  event.preventDefault();
  beginViewerCompareHold();
});
viewerMetaEl.addEventListener("keyup", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (!target.closest(".viewer-compare-hold")) return;
  if (event.key !== " " && event.key !== "Enter") return;
  event.preventDefault();
  endViewerCompareHold();
});
viewerMetaEl.addEventListener(
  "blur",
  (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (!target.closest(".viewer-compare-hold")) return;
    endViewerCompareHold();
  },
  true
);
viewerModalEl.addEventListener("click", (event) => {
  if (event.target === viewerModalEl) {
    hideViewer();
  }
});
viewerStageEl.addEventListener("pointerdown", beginDrag);
viewerStageEl.addEventListener("pointermove", moveDrag);
viewerStageEl.addEventListener("pointerup", endDrag);
viewerStageEl.addEventListener("pointercancel", endDrag);
viewerStageEl.addEventListener("pointerleave", (event) => {
  if (!state.dragging) return;
  if (typeof event.buttons === "number" && event.buttons !== 0) return;
  endDrag(event);
});
window.addEventListener("blur", () => endDrag());
document.addEventListener("pointerup", () => endViewerCompareHold());
viewerStageEl.addEventListener("dragstart", (event) => event.preventDefault());
viewerImageEl.setAttribute("draggable", "false");
viewerStageEl.addEventListener(
  "wheel",
  (event) => {
    if (viewerModalEl.classList.contains("hidden")) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.1 : 0.9;
    setZoom(state.zoom * factor);
  },
  { passive: false }
);
viewerStageEl.addEventListener("dblclick", () => {
  state.panX = 0;
  state.panY = 0;
  setZoom(1.0);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !zipProgressModalEl.classList.contains("hidden")) {
    event.preventDefault();
    cancelZipDownload();
    return;
  }
  const viewerOpen = !viewerModalEl.classList.contains("hidden");
  if (viewerOpen && event.key === "ArrowLeft") {
    event.preventDefault();
    showViewerByOffset(-1);
    return;
  }
  if (viewerOpen && event.key === "ArrowRight") {
    event.preventDefault();
    showViewerByOffset(1);
    return;
  }
  if (viewerOpen && (event.key === "Delete" || event.key === "Backspace")) {
    event.preventDefault();
    const item = getActiveViewerItem();
    if (!item) return;
    showConfirmModal(
      `Delete "${item.filename}"? This cannot be undone.`,
      async () => {
        await deleteImage(item.filename);
      },
      "Yes",
      "No"
    );
    return;
  }
  if (event.key === "Escape") {
    hideConfirmModal();
    hideViewer();
    setSettingsVisible(false);
    if (state.multiSelectMode) {
      toggleMultiSelectMode(false);
    }
  }
});

confirmModalCancelEl.addEventListener("click", hideConfirmModal);
confirmModalEl.addEventListener("click", (event) => {
  if (event.target === confirmModalEl) {
    hideConfirmModal();
  }
});
zipProgressCancelEl.addEventListener("click", cancelZipDownload);
zipProgressModalEl.addEventListener("click", (event) => {
  if (event.target === zipProgressModalEl) {
    cancelZipDownload();
  }
});
confirmModalConfirmEl.addEventListener("click", async () => {
  const action = state.confirmAction;
  hideConfirmModal();
  if (!action) return;
  try {
    await action();
  } catch (error) {
    setStatus(String(error?.message || error), true);
  }
});

bootstrap();
