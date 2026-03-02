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
const schedulerSamplerButtonEl = document.getElementById("scheduler-sampler-button");
const promptEnhanceButtonEl = document.getElementById("prompt-enhance-button");
const deleteGalleryButtonEl = document.getElementById("delete-gallery-button");
const killServerButtonEl = document.getElementById("kill-server-button");
const filterInputEl = document.getElementById("filter-input");
const reverseOrderButtonEl = document.getElementById("reverse-order-button");
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
  ["scheduler-sampler-button", schedulerSamplerButtonEl],
  ["prompt-enhance-button", promptEnhanceButtonEl],
  ["delete-gallery-button", deleteGalleryButtonEl],
  ["kill-server-button", killServerButtonEl],
  ["filter-input", filterInputEl],
  ["reverse-order-button", reverseOrderButtonEl],
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
  ["disconnect-overlay", disconnectOverlayEl],
];

const missingUi = requiredUi.filter(([, element]) => !element).map(([name]) => name);
if (missingUi.length) {
  throw new Error(`UI initialization failed. Missing element(s): ${missingUi.join(", ")}`);
}

const state = {
  orientation: "portrait",
  freezeSeed: false,
  dpmSampler: false,
  promptEnhance: true,
  currentSeed: null,
  newestFirst: true,
  filterTimer: null,
  maxQueuedGenerations: 5,
  queue: [],
  activeJob: null,
  queueWorkerRunning: false,
  galleryItems: [],
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
  const generateShift = promptTop - buttonTop;
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
  return `/images/${encodeURIComponent(filename)}?t=${Date.now()}`;
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
  if (state.dpmSampler) {
    pieces.push(`Scheduler <span class="summary-value">DPM</span>`);
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

function updateFreezeSeedButton() {
  if (state.freezeSeed) {
    freezeSeedButtonEl.textContent = `FREEZE SEED: ON (${state.currentSeed})`;
    freezeSeedButtonEl.classList.add("active");
  } else {
    freezeSeedButtonEl.textContent = "FREEZE SEED: OFF";
    freezeSeedButtonEl.classList.remove("active");
  }
}

function updateSchedulerSamplerButton() {
  if (state.dpmSampler) {
    schedulerSamplerButtonEl.textContent = "SCHEDULER/SAMPLER: ON (DPM)";
    schedulerSamplerButtonEl.classList.add("active");
  } else {
    schedulerSamplerButtonEl.textContent = "SCHEDULER/SAMPLER: OFF (EULER)";
    schedulerSamplerButtonEl.classList.remove("active");
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
  viewerImageEl.src = buildImageUrl(item.filename);
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
  viewerImageEl.src = buildImageUrl(sourceFilename);
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
  viewerDownloadEl.href = `/images/${encodeURIComponent(item.filename)}`;
  viewerDownloadEl.setAttribute("download", item.filename);
}

function showViewer(item, index = -1) {
  if (index < 0) {
    index = state.galleryItems.findIndex((candidate) => candidate.filename === item.filename);
  }
  state.viewerIndex = index;
  state.viewerCompareHolding = false;
  state.viewerCompareSourceFilename = null;
  viewerImageEl.src = buildImageUrl(item.filename);
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

function findPendingJob(placeholderId) {
  return state.pendingJobs.find((job) => job.placeholderId === placeholderId) || null;
}

function removePendingJob(placeholderId) {
  state.pendingJobs = state.pendingJobs.filter((job) => job.placeholderId !== placeholderId);
}

function pendingJobLabel(job, queuePosition) {
  const isUpscale = job.kind === "upscale";
  if (job.status === "generating") {
    return isUpscale ? "UPSCALING..." : "GENERATING...";
  }
  if (queuePosition >= 0) {
    const prefix = isUpscale ? "UPSCALE QUEUED" : "QUEUED";
    return `${prefix} (${queuePosition + 1})`;
  }
  return isUpscale ? "UPSCALE QUEUED..." : "QUEUED...";
}

function renderPendingTile(job) {
  const tile = document.createElement("article");
  tile.className = "tile generating";
  tile.dataset.placeholderId = job.placeholderId;

  const canvas = document.createElement("div");
  canvas.className = "tile-placeholder";
  canvas.style.aspectRatio = `${job.width} / ${job.height}`;

  const spinner = document.createElement("div");
  spinner.className = "tile-spinner";
  const queuePosition = state.queue.findIndex((queued) => queued.placeholderId === job.placeholderId);
  spinner.textContent = pendingJobLabel(job, queuePosition);

  canvas.append(spinner);
  tile.append(canvas);
  return tile;
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

function renderImageTile(item, index) {
  const tile = document.createElement("article");
  tile.className = "tile";

  const image = document.createElement("img");
  image.src = `/images/${encodeURIComponent(item.filename)}?t=${Date.now()}`;
  image.alt = item.prompt || "Generated image";
  image.loading = "lazy";
  image.addEventListener("error", () => {
    tile.remove();
    dropMissingGalleryItem(item.filename);
  });

  const overlay = document.createElement("div");
  overlay.className = "tile-overlay";

  const meta = document.createElement("div");
  meta.className = "tile-meta";
  const timestamp = item.timestamp || item.created_at;
  const resolution = item.width && item.height ? `${item.width}x${item.height}` : "unknown";
  meta.textContent = `${formatGalleryTimestamp(timestamp)} | ${shortPrompt(item.prompt, 60)} | ${resolution}`;

  const actions = document.createElement("div");
  actions.className = "tile-actions";
  const primaryActions = document.createElement("div");
  primaryActions.className = "tile-primary-actions";

  const download = document.createElement("a");
  download.className = "tile-download";
  download.href = `/images/${encodeURIComponent(item.filename)}`;
  download.setAttribute("download", item.filename);
  download.textContent = "Download";
  download.addEventListener("click", (event) => event.stopPropagation());

  const del = document.createElement("button");
  del.className = "tile-delete";
  del.type = "button";
  del.setAttribute("aria-label", `Delete ${item.filename}`);
  del.title = "Delete image";
  del.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l1 2h5v2H3V5h5l1-2zm1 7h2v8h-2v-8zm4 0h2v8h-2v-8zM7 10h2v8H7v-8z"/></svg>';
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    showConfirmModal(`Delete "${item.filename}"? This cannot be undone.`, async () => {
      await deleteImage(item.filename);
    }, "Delete");
  });

  primaryActions.append(download);
  if (canUpscaleItem(item)) {
    const upscale = document.createElement("button");
    upscale.className = "tile-upscale";
    upscale.type = "button";
    upscale.textContent = "Upscale";
    upscale.addEventListener("click", (event) => {
      event.stopPropagation();
      enqueueUpscaleFromItem(item);
    });
    primaryActions.append(upscale);
  }
  actions.append(primaryActions, del);
  overlay.append(meta, actions);
  tile.append(image, overlay);
  tile.addEventListener("click", () => showViewer(item, index));
  return tile;
}

function renderGallery() {
  galleryEl.innerHTML = "";
  const pending = [...state.pendingJobs];
  const items = [...state.galleryItems];
  const hasContent = pending.length > 0 || items.length > 0;
  emptyStateEl.classList.toggle("hidden", hasContent);
  if (!hasContent) return;

  if (state.newestFirst) {
    for (const job of pending) {
      galleryEl.append(renderPendingTile(job));
    }
    for (let index = 0; index < items.length; index += 1) {
      galleryEl.append(renderImageTile(items[index], index));
    }
    return;
  }

  for (let index = 0; index < items.length; index += 1) {
    galleryEl.append(renderImageTile(items[index], index));
  }
  for (const job of pending) {
    galleryEl.append(renderPendingTile(job));
  }
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

  const response = await fetch(`/images?${query.toString()}`, { cache: "no-store" });
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
  state.galleryItems = sortItems(payload.items || []);
  renderGallery();
  syncViewerWithGallery();
}

async function deleteImage(filename) {
  const existingItems = state.galleryItems;
  const hadItem = existingItems.some((item) => item.filename === filename);
  if (hadItem) {
    state.galleryItems = existingItems.filter((item) => item.filename !== filename);
    if (state.viewerFilename === filename) {
      hideViewer();
    }
    renderGallery();
    syncViewerWithGallery();
  }

  let payload = null;
  const response = await fetch(`/images/${encodeURIComponent(filename)}?confirm=DELETE`, {
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
  await loadImages();
  setStatus(`Deleted ${filename}.`);
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
    scheduler_mode: state.dpmSampler ? "dpm" : "euler",
    enhance_prompt: state.promptEnhance,
  };

  state.queue.push(job);
  state.pendingJobs.push({
    kind: "generate",
    placeholderId,
    width: dimensions.width,
    height: dimensions.height,
    status: "queued",
  });
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
    scheduler_mode: state.dpmSampler ? "dpm" : "euler",
    enhance_prompt: state.promptEnhance,
  };

  state.queue.push(job);
  state.pendingJobs.push({
    kind: "upscale",
    placeholderId,
    width: targetWidth,
    height: targetHeight,
    status: "queued",
  });
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
      if (!state.activeJob) {
        state.activeJob = state.queue.shift() || null;
        if (!state.activeJob) break;
        const pending = findPendingJob(state.activeJob.placeholderId);
        if (pending) {
          pending.status = "generating";
        }
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
              filename: job.filename,
              pack: job.pack,
              seed: job.seed,
              scheduler_mode: job.scheduler_mode,
              enhance_prompt: job.enhance_prompt,
            }
          : {
              prompt: job.prompt,
              width: job.width,
              height: job.height,
              seed: job.seed,
              scheduler_mode: job.scheduler_mode,
              enhance_prompt: job.enhance_prompt,
            };
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payloadBody),
        });
        const payload = await response.json();
        if (!response.ok) {
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
        removePendingJob(job.placeholderId);
        state.activeJob = null;
        await loadImages();
      } catch (error) {
        removePendingJob(job.placeholderId);
        state.activeJob = null;
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
    const response = await fetch(`/gallery?confirm=${encoded}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: confirmation }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(payload, "Gallery deletion failed."));
    }
    hideViewer();
    galleryEl.innerHTML = "";
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
    const response = await fetch("/server/kill", {
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

function toggleSchedulerSampler() {
  state.dpmSampler = !state.dpmSampler;
  updateSchedulerSamplerButton();
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
    updateTopbarOffset();
    updateReverseButton();
    applyOrientationButtonState();
    updateFreezeSeedButton();
    updateSchedulerSamplerButton();
    updatePromptEnhanceButton();
    updateSettingsSummary();
    updateViewerNavState();
    updateGenerateButtonState();
    await loadImages();
    setStatus("Ready.");
  } catch (error) {
    setStatus(String(error?.message || error), true);
  }
}

settingsButtonEl.addEventListener("click", toggleSettingsVisible);
generateButtonEl.addEventListener("click", () => {
  enqueueGenerationFromPrompt();
});
freezeSeedButtonEl.addEventListener("click", toggleFreezeSeed);
schedulerSamplerButtonEl.addEventListener("click", toggleSchedulerSampler);
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
window.addEventListener("resize", updateTopbarOffset);
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
  }
});

confirmModalCancelEl.addEventListener("click", hideConfirmModal);
confirmModalEl.addEventListener("click", (event) => {
  if (event.target === confirmModalEl) {
    hideConfirmModal();
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
