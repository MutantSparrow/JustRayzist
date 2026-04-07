const topbarEl = document.querySelector(".topbar");
const promptInputEl = document.getElementById("prompt-input");
const generateButtonEl = document.getElementById("generate-button");
const generateButtonLabelEl = document.getElementById("generate-button-label");
const loraDrawerToggleEl = document.getElementById("lora-drawer-toggle");
const settingsButtonEl = document.getElementById("settings-button");
const settingsPanelEl = document.getElementById("settings-panel");
const settingsSummaryEl = document.getElementById("settings-summary");
const resolutionSelectEl = document.getElementById("resolution-select");
const orientationToggleEl = document.getElementById("orientation-toggle");
const freezeSeedButtonEl = document.getElementById("freeze-seed-button");
const proceduralLatentSliderEl = document.getElementById("procedural-latent-slider");
const proceduralLatentValueEl = document.getElementById("procedural-latent-value");
const promptEnhanceButtonEl = document.getElementById("prompt-enhance-button");
const upscaleModeToggleEl = document.getElementById("upscale-mode-toggle");
const upscaleScaleToggleEl = document.getElementById("upscale-scale-toggle");
const upscaleSettingsHintEl = document.getElementById("upscale-settings-hint");
const deleteGalleryButtonEl = document.getElementById("delete-gallery-button");
const killServerButtonEl = document.getElementById("kill-server-button");
const filterInputEl = document.getElementById("filter-input");
const reverseOrderButtonEl = document.getElementById("reverse-order-button");
const galleryColorFiltersEl = document.getElementById("gallery-color-filters");
const galleryFavoriteFilterButtonEl = document.getElementById("gallery-favorite-filter");
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
const loraDrawerBackdropEl = document.getElementById("lora-drawer-backdrop");
const loraDrawerEl = document.getElementById("lora-drawer");
const loraDrawerCloseEl = document.getElementById("lora-drawer-close");
const loraDrawerCapabilityEl = document.getElementById("lora-drawer-capability");
const loraFilterInputEl = document.getElementById("lora-filter-input");
const loraUploadInputEl = document.getElementById("lora-upload-input");
const loraDrawerEmptyEl = document.getElementById("lora-drawer-empty");
const loraListEl = document.getElementById("lora-list");
const loraEditorModalEl = document.getElementById("lora-editor-modal");
const loraEditorCloseEl = document.getElementById("lora-editor-close");
const loraEditorTitleEl = document.getElementById("lora-editor-title");
const loraEditorFileEl = document.getElementById("lora-editor-file");
const loraEditorProgressEl = document.getElementById("lora-editor-progress");
const loraEditorProgressSpinnerEl = document.getElementById("lora-editor-progress-spinner");
const loraEditorBodyEl = loraEditorModalEl?.querySelector(".lora-editor-body");
const loraEditorNameEl = document.getElementById("lora-editor-name");
const loraEditorThumbnailPreviewEl = document.getElementById("lora-editor-thumbnail-preview");
const loraEditorThumbnailButtonEl = document.getElementById("lora-editor-thumbnail-button");
const loraEditorThumbnailInputEl = document.getElementById("lora-editor-thumbnail-input");
const loraEditorTriggerChipsEl = document.getElementById("lora-editor-trigger-chips");
const loraEditorTriggerInputEl = document.getElementById("lora-editor-trigger-input");
const loraEditorSaveEl = document.getElementById("lora-editor-save");
const viewerModalEl = document.getElementById("viewer-modal");
const viewerMetaEl = document.getElementById("viewer-meta");
const viewerImageEl = document.getElementById("viewer-image");
const viewerDownloadEl = document.getElementById("viewer-download");
const viewerStageEl = document.getElementById("viewer-stage");
const viewerStageFxEl = document.getElementById("viewer-stage-fx");
const viewerStageFxLabelEl = document.getElementById("viewer-stage-fx-label");
const viewerCloseButtonEl = document.getElementById("viewer-close-button");
const viewerDeleteButtonEl = document.getElementById("viewer-delete-button");
const viewerUsePromptButtonEl = document.getElementById("viewer-use-prompt-button");
const viewerCopyPromptButtonEl = document.getElementById("viewer-copy-prompt-button");
const viewerUpscaleButtonEl = document.getElementById("viewer-upscale-button");
const viewerFavoriteButtonEl = document.getElementById("viewer-favorite-button");
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
  ["lora-drawer-toggle", loraDrawerToggleEl],
  ["settings-button", settingsButtonEl],
  ["settings-panel", settingsPanelEl],
  ["settings-summary", settingsSummaryEl],
  ["resolution-select", resolutionSelectEl],
  ["orientation-toggle", orientationToggleEl],
  ["freeze-seed-button", freezeSeedButtonEl],
  ["procedural-latent-slider", proceduralLatentSliderEl],
  ["procedural-latent-value", proceduralLatentValueEl],
  ["prompt-enhance-button", promptEnhanceButtonEl],
  ["upscale-settings-hint", upscaleSettingsHintEl],
  ["delete-gallery-button", deleteGalleryButtonEl],
  ["kill-server-button", killServerButtonEl],
  ["filter-input", filterInputEl],
  ["reverse-order-button", reverseOrderButtonEl],
  ["gallery-color-filters", galleryColorFiltersEl],
  ["gallery-favorite-filter", galleryFavoriteFilterButtonEl],
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
  ["lora-drawer-backdrop", loraDrawerBackdropEl],
  ["lora-drawer", loraDrawerEl],
  ["lora-drawer-close", loraDrawerCloseEl],
  ["lora-drawer-capability", loraDrawerCapabilityEl],
  ["lora-filter-input", loraFilterInputEl],
  ["lora-upload-input", loraUploadInputEl],
  ["lora-drawer-empty", loraDrawerEmptyEl],
  ["lora-list", loraListEl],
  ["lora-editor-modal", loraEditorModalEl],
  ["lora-editor-close", loraEditorCloseEl],
  ["lora-editor-title", loraEditorTitleEl],
  ["lora-editor-file", loraEditorFileEl],
  ["lora-editor-progress", loraEditorProgressEl],
  ["lora-editor-progress-spinner", loraEditorProgressSpinnerEl],
  ["lora-editor-body", loraEditorBodyEl],
  ["lora-editor-name", loraEditorNameEl],
  ["lora-editor-thumbnail-preview", loraEditorThumbnailPreviewEl],
  ["lora-editor-thumbnail-button", loraEditorThumbnailButtonEl],
  ["lora-editor-thumbnail-input", loraEditorThumbnailInputEl],
  ["lora-editor-trigger-chips", loraEditorTriggerChipsEl],
  ["lora-editor-trigger-input", loraEditorTriggerInputEl],
  ["lora-editor-save", loraEditorSaveEl],
  ["viewer-modal", viewerModalEl],
  ["viewer-meta", viewerMetaEl],
  ["viewer-image", viewerImageEl],
  ["viewer-download", viewerDownloadEl],
  ["viewer-stage", viewerStageEl],
  ["viewer-stage-fx", viewerStageFxEl],
  ["viewer-stage-fx-label", viewerStageFxLabelEl],
  ["viewer-close-button", viewerCloseButtonEl],
  ["viewer-delete-button", viewerDeleteButtonEl],
  ["viewer-use-prompt-button", viewerUsePromptButtonEl],
  ["viewer-copy-prompt-button", viewerCopyPromptButtonEl],
  ["viewer-upscale-button", viewerUpscaleButtonEl],
  ["viewer-favorite-button", viewerFavoriteButtonEl],
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
const LORA_APPLIED_STORAGE_KEY = "justrayzist.lora_applied";
const CLIENT_JOB_POLL_INTERVAL_MS = 1500;
const CLIENT_QUEUE_STORAGE_VERSION = 2;
const GALLERY_COLOR_FILTERS = ["black", "white", "red", "yellow", "blue", "green"];
const GALLERY_COLOR_CACHE_STATUS_MESSAGE = "Updating gallery color cache...";
const GALLERY_COLOR_CACHE_POLL_INTERVAL_MS = 2500;
const LORA_LIBRARY_POLL_INTERVAL_MS = 2500;
const LORA_MASONRY_SINGLE_COLUMN_BREAKPOINT_PX = 600;
const GALLERY_SOFT_REMOVAL_DURATION_MS = 240;
const TILE_ACTION_FX_DURATION_MS = 190;
const PENDING_UPSCALE_ENTRY_FX_DURATION_MS = 220;
const VIEWER_STAGE_FX_DURATION_MS = 210;
const DEFAULT_LORA_WEIGHT = 1.0;
const MAX_ACTIVE_LORAS = 3;
const MIN_LORA_WEIGHT = -2.0;
const MAX_LORA_WEIGHT = 2.0;
const LORA_STRENGTH_PRESETS = (() => {
  const presets = [];
  for (let value = MIN_LORA_WEIGHT; value <= MAX_LORA_WEIGHT + 0.0001; value += 0.25) {
    const rounded = Math.round(value * 100) / 100;
    if (Math.abs(rounded) < 0.0001) continue;
    presets.push({ value: rounded });
  }
  return presets;
})();

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

function normalizeLoraWeight(rawValue) {
  const value = Number(rawValue);
  if (!Number.isFinite(value)) return DEFAULT_LORA_WEIGHT;
  if (Math.abs(value) < 0.0001) return DEFAULT_LORA_WEIGHT;
  let best = LORA_STRENGTH_PRESETS[0];
  for (const preset of LORA_STRENGTH_PRESETS) {
    if (Math.abs(preset.value - value) < Math.abs(best.value - value)) {
      best = preset;
    }
  }
  return best.value;
}

function normalizeLoraSelections(rawSelections) {
  if (!Array.isArray(rawSelections)) return [];
  const normalized = [];
  const seen = new Set();
  for (const item of rawSelections) {
    const id = String(item?.id || "").trim();
    if (!id || seen.has(id)) continue;
    normalized.push({ id, weight: normalizeLoraWeight(item?.weight) });
    seen.add(id);
    if (normalized.length >= MAX_ACTIVE_LORAS) break;
  }
  return normalized;
}

function loadAppliedLorasFromSession() {
  try {
    const raw = window.sessionStorage.getItem(LORA_APPLIED_STORAGE_KEY);
    if (!raw) return [];
    return normalizeLoraSelections(JSON.parse(raw));
  } catch (_) {
    return [];
  }
}

const state = {
  clientId: getOrCreateClientId(),
  orientation: "portrait",
  freezeSeed: false,
  proceduralCreativity: 0,
  promptEnhance: true,
  upscaleMode: "fast",
  upscaleScale: 2,
  galleryColumns: getStoredGalleryColumns(),
  currentSeed: null,
  newestFirst: true,
  activeColorFilter: null,
  favoritesOnly: false,
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
  loraLibraryLoadRequestSeq: 0,
  loraLibrarySignature: "",
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
  favoriteToneByFilename: new Map(),
  lastGalleryQueryState: null,
  galleryRenderRevision: 0,
  gallerySoftRemovalBatches: new Map(),
  pendingUpscaleFxIds: new Set(),
  viewerStageFxTimer: null,
  viewerStageFxResolve: null,
  notificationPermissionRequested: false,
  loraDrawerOpen: false,
  loraLibrary: [],
  loraFilter: "",
  loraLibraryMasonryFrame: null,
  loraLibraryPollTimer: null,
  loraLibraryEventSource: null,
  loraLibraryEventReconnectTimer: null,
  loraCapabilities: {
    supported: false,
    max_active: MAX_ACTIVE_LORAS,
    min_weight: MIN_LORA_WEIGHT,
    max_weight: MAX_LORA_WEIGHT,
    default_weight: DEFAULT_LORA_WEIGHT,
  },
  loraPendingSelections: [],
  loraAppliedSelections: loadAppliedLorasFromSession(),
  loraUploadBusy: false,
  loraSaveBusy: false,
  loraUploadRequest: null,
  loraTouchFocusId: null,
  loraEditorOpen: false,
  loraEditorPhase: "editing",
  loraEditorUploadProgress: 0,
  loraEditorMode: "create",
  loraEditorDraftId: null,
  loraEditorLoraId: null,
  loraEditorSourceFilename: "",
  loraEditorDisplayName: "",
  loraEditorTriggerWords: [],
  loraEditorDetectedTriggerWords: [],
  loraEditorMetadataSummary: {},
  loraEditorPreviewUrl: "",
  loraEditorPreviewObjectUrl: null,
  loraEditorPreviewBlob: null,
  loraEditorBusy: false,
};

function randomSeed() {
  return Math.floor(Math.random() * 2_147_483_646) + 1;
}

function updateTopbarOffset() {
  const offset = topbarEl.offsetHeight;
  const topbarRect = topbarEl.getBoundingClientRect();
  const promptRect = promptInputEl.getBoundingClientRect();
  const promptTop = Math.max(0, Math.round(promptRect.top - topbarRect.top));
  const promptRight = Math.max(0, Math.round(promptRect.right - topbarRect.left));
  const generateLeft = window.innerWidth <= 960 ? 0 : promptRight + 8;
  document.documentElement.style.setProperty("--topbar-offset", `${offset}px`);
  document.documentElement.style.setProperty("--generate-shift", "0px");
  document.documentElement.style.setProperty("--generate-controls-top", `${promptTop}px`);
  document.documentElement.style.setProperty("--generate-controls-left", `${generateLeft}px`);
  if (isSettingsOpen()) {
    positionSettingsPanel();
  }
}

function setStatus(message, isError = false) {
  statusLineEl.textContent = String(message || "");
  statusLineEl.classList.toggle("error", Boolean(isError));
}

function supportsBrowserNotifications() {
  return typeof window !== "undefined" && "Notification" in window && window.isSecureContext !== false;
}

function requestCompletionNotificationPermission() {
  if (!supportsBrowserNotifications()) return;
  if (state.notificationPermissionRequested) return;
  if (Notification.permission !== "default") return;
  state.notificationPermissionRequested = true;
  Promise.resolve(Notification.requestPermission()).catch(() => {
  });
}

function showGenerationCompletionNotification(job, payload) {
  if (!supportsBrowserNotifications()) return false;
  if (Notification.permission !== "granted") return false;
  if (!job || job.kind !== "generate") return false;

  const filename = String(payload?.filename || "").trim();
  const seed = payload?.seed ?? job.seed ?? null;
  const duration = Number(payload?.duration_ms);
  const title = payload?.prompt_enhanced ? "Prompt Enhanced" : "Generation Done";
  const details = [];
  if (filename) {
    details.push(filename);
  } else {
    details.push("Image saved");
  }
  if (Number.isFinite(duration) && duration > 0) {
    details.push(`${duration} ms`);
  }
  if (seed !== null && seed !== undefined && seed !== "") {
    details.push(`seed ${seed}`);
  }
  const body = details.join(" | ");
  const promptSnippet = String(job.prompt || "").trim();
  const imageUrl = filename ? new URL(buildImageUrl(filename), window.location.origin).toString() : undefined;
  const notification = new Notification(title, {
    body: promptSnippet ? `${body}\n${shortPrompt(promptSnippet, 96)}` : body,
    icon: new URL("/img/favicon.ico", window.location.origin).toString(),
    image: imageUrl,
    tag: filename ? `generation:${filename}` : `generation:${job.placeholderId}`,
  });
  notification.onclick = () => {
    try {
      window.focus();
    } catch (_) {
    }
    notification.close();
  };
  window.setTimeout(() => notification.close(), 10000);
  return true;
}

function getTileActionFxClassName(kind) {
  if (kind === "delete") return "fx-delete";
  if (kind === "upscale-source") return "fx-upscale-source";
  if (kind === "pending-upscale-enter") return "pending-upscale-enter";
  return "";
}

function clearTileActionFx(tile, options = {}) {
  if (!(tile instanceof HTMLElement)) return;
  const onlyKind = String(options.onlyKind || "").trim();
  const activeKind = String(tile._tileActionFxKind || "");
  if (onlyKind && activeKind !== onlyKind) {
    return;
  }
  if (tile._tileActionFxTimer !== null && tile._tileActionFxTimer !== undefined) {
    window.clearTimeout(tile._tileActionFxTimer);
  }
  tile._tileActionFxTimer = null;
  tile._tileActionFxKind = "";
  tile.classList.remove("fx-delete", "fx-upscale-source", "pending-upscale-enter");
}

function playTileActionFxOnTile(tile, kind, options = {}) {
  const className = getTileActionFxClassName(kind);
  if (!(tile instanceof HTMLElement) || !className) return false;
  const duration = Math.max(
    1,
    Number(
      options.duration ??
        (kind === "pending-upscale-enter"
          ? PENDING_UPSCALE_ENTRY_FX_DURATION_MS
          : TILE_ACTION_FX_DURATION_MS),
    ) || TILE_ACTION_FX_DURATION_MS,
  );
  clearTileActionFx(tile);
  void tile.offsetWidth;
  tile._tileActionFxKind = kind;
  tile.classList.add(className);
  const timeoutId = window.setTimeout(() => {
    if (tile._tileActionFxTimer !== timeoutId) return;
    clearTileActionFx(tile);
  }, duration);
  tile._tileActionFxTimer = timeoutId;
  return true;
}

function playTileActionFx(filename, kind, options = {}) {
  return playTileActionFxOnTile(getGalleryTileByFilename(filename), kind, options);
}

function consumePendingUpscaleEntryFx(placeholderId) {
  const target = String(placeholderId || "").trim();
  if (!target || !state.pendingUpscaleFxIds.has(target)) return false;
  state.pendingUpscaleFxIds.delete(target);
  return true;
}

function clearViewerActionFx(options = {}) {
  const resolvePending = options.resolvePending !== false;
  if (state.viewerStageFxTimer !== null) {
    window.clearTimeout(state.viewerStageFxTimer);
    state.viewerStageFxTimer = null;
  }
  const resolve = state.viewerStageFxResolve;
  state.viewerStageFxResolve = null;
  viewerStageFxEl.classList.remove("active", "fx-delete", "fx-upscale");
  viewerStageFxLabelEl.textContent = "";
  if (resolvePending && typeof resolve === "function") {
    resolve();
  }
}

function playViewerActionFx(kind) {
  if (viewerModalEl.classList.contains("hidden")) {
    return Promise.resolve(false);
  }
  clearViewerActionFx();
  const fxKind = kind === "delete" ? "delete" : "upscale";
  viewerStageFxLabelEl.textContent = fxKind === "delete" ? "DEL" : "2x";
  viewerStageFxEl.classList.add(fxKind === "delete" ? "fx-delete" : "fx-upscale");
  void viewerStageFxEl.offsetWidth;
  viewerStageFxEl.classList.add("active");
  return new Promise((resolve) => {
    state.viewerStageFxResolve = () => resolve(true);
    state.viewerStageFxTimer = window.setTimeout(() => {
      clearViewerActionFx();
    }, VIEWER_STAGE_FX_DURATION_MS);
  });
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

function cloneLoraSelections(selections) {
  return normalizeLoraSelections(selections).map((item) => ({ ...item }));
}

function loraStrengthPresetForWeight(weight) {
  const normalized = normalizeLoraWeight(weight);
  return (
    LORA_STRENGTH_PRESETS.find((item) => item.value === normalized)
    || LORA_STRENGTH_PRESETS.find((item) => item.value === DEFAULT_LORA_WEIGHT)
    || LORA_STRENGTH_PRESETS[0]
  );
}

function loraStrengthPresetIndex(weight) {
  const normalized = normalizeLoraWeight(weight);
  const index = LORA_STRENGTH_PRESETS.findIndex((item) => item.value === normalized);
  if (index >= 0) return index;
  const defaultIndex = LORA_STRENGTH_PRESETS.findIndex((item) => item.value === DEFAULT_LORA_WEIGHT);
  return defaultIndex >= 0 ? defaultIndex : 0;
}

function loraStrengthPresetAt(index) {
  const resolved = Math.max(0, Math.min(LORA_STRENGTH_PRESETS.length - 1, Number(index) || 0));
  return LORA_STRENGTH_PRESETS[resolved];
}

function isTouchLikeLoraUi() {
  return Boolean(window.matchMedia && window.matchMedia("(hover: none), (pointer: coarse)").matches);
}

function formatInitialLoraDisplayName(filename) {
  const stem = String(filename || "").replace(/\.[^.]+$/, "").trim();
  if (!stem) return "";
  return stem
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function loraEditorIsUploading() {
  return state.loraEditorPhase === "uploading" || state.loraEditorPhase === "analyzing";
}

function loraEditorIsEditing() {
  return state.loraEditorPhase === "editing";
}

function loraSelectionsEqual(left, right) {
  const normalizedLeft = normalizeLoraSelections(left);
  const normalizedRight = normalizeLoraSelections(right);
  if (normalizedLeft.length !== normalizedRight.length) return false;
  return normalizedLeft.every((item, index) => {
    const other = normalizedRight[index];
    return item.id === other.id && Math.abs(item.weight - other.weight) < 0.0001;
  });
}

function persistAppliedLorasToSession() {
  try {
    window.sessionStorage.setItem(LORA_APPLIED_STORAGE_KEY, JSON.stringify(state.loraAppliedSelections));
  } catch (_) {
  }
}

function getLoraById(loraId) {
  const target = String(loraId || "").trim();
  if (!target) return null;
  return state.loraLibrary.find((item) => item.id === target) || null;
}

function sanitizeLoraSelectionsAgainstLibrary(selections) {
  const availableIds = new Set(state.loraLibrary.map((item) => item.id));
  return normalizeLoraSelections(selections).filter((item) => availableIds.has(item.id));
}

function buildLoraPreviewUrl(lora) {
  const previewCacheKey = String(lora?.preview_cache_key || lora?.updated_at || "").trim();
  const query = new URLSearchParams();
  if (previewCacheKey) {
    query.set("t", previewCacheKey);
  }
  const suffix = query.toString();
  return `/loras/${encodeURIComponent(lora.id)}/preview${suffix ? `?${suffix}` : ""}`;
}

function appliedLoraCount() {
  return state.loraAppliedSelections.length;
}

function pendingLoraCount() {
  return state.loraPendingSelections.length;
}

function loraSelectionsDirty() {
  return !loraSelectionsEqual(state.loraPendingSelections, state.loraAppliedSelections);
}

function setLoraDrawerOpen(open) {
  const next = Boolean(open);
  const wasOpen = state.loraDrawerOpen;
  const mobileOverlay = window.innerWidth <= 960;
  state.loraDrawerOpen = next;
  document.body.classList.toggle("lora-drawer-open", next && !mobileOverlay);
  loraDrawerEl.classList.toggle("open", next);
  loraDrawerEl.setAttribute("aria-hidden", String(!next));
  loraDrawerBackdropEl.classList.toggle("hidden", !next || !mobileOverlay);
  loraDrawerBackdropEl.setAttribute("aria-hidden", String(!next || !mobileOverlay));
  loraDrawerToggleEl.setAttribute("aria-expanded", String(next));
  if (!next) {
    state.loraTouchFocusId = null;
    stopLoraLibraryPolling();
    if (state.loraLibraryMasonryFrame !== null) {
      window.cancelAnimationFrame(state.loraLibraryMasonryFrame);
      state.loraLibraryMasonryFrame = null;
    }
  }
  updateTopbarOffset();
  if (next) {
    loraFilterInputEl.focus();
    if (!wasOpen) {
      loadLoraLibrary({ refreshSummary: false, silent: true }).catch(() => {
      });
    }
    scheduleLoraLibraryPoll();
    scheduleLoraLibraryMasonryRelayout();
  }
}

function formatLoraWeight(value) {
  return normalizeLoraWeight(value).toFixed(2);
}

function formatLoraPresetLabel(value) {
  const preset = loraStrengthPresetForWeight(value);
  return preset.value > 0 ? `+${preset.value.toFixed(2)}` : preset.value.toFixed(2);
}

function parseImageLoras(item) {
  if (Array.isArray(item?.loras)) {
    return item.loras;
  }
  const raw = String(item?.loras_json || "").trim();
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function updateLoraCapabilityUi() {
  const supported = Boolean(state.loraCapabilities?.supported);
  loraDrawerToggleEl.disabled = !supported;
  loraDrawerCloseEl.disabled = state.loraSaveBusy;
  loraDrawerCapabilityEl.classList.toggle("error", !supported);
  if (supported) {
    loraDrawerCapabilityEl.textContent = "";
  } else {
    loraDrawerCapabilityEl.textContent = "LoRAs are unavailable for the current runtime.";
  }
}

function cleanupLoraEditorPreviewObjectUrl() {
  if (state.loraEditorPreviewObjectUrl) {
    URL.revokeObjectURL(state.loraEditorPreviewObjectUrl);
    state.loraEditorPreviewObjectUrl = null;
  }
}

function setLoraEditorPreview(previewUrl, previewBlob = null) {
  cleanupLoraEditorPreviewObjectUrl();
  state.loraEditorPreviewBlob = previewBlob || null;
  state.loraEditorPreviewUrl = String(previewUrl || "").trim();
  if (state.loraEditorPreviewUrl && previewBlob) {
    state.loraEditorPreviewObjectUrl = state.loraEditorPreviewUrl;
  }
}

function renderLoraEditorTriggerChips() {
  loraEditorTriggerChipsEl.innerHTML = "";
  state.loraEditorTriggerWords.forEach((triggerWord) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "lora-trigger-chip";
    chip.textContent = triggerWord;
    chip.title = `Remove ${triggerWord}`;
    chip.addEventListener("click", () => {
      state.loraEditorTriggerWords = state.loraEditorTriggerWords.filter((item) => item !== triggerWord);
      renderLoraEditorTriggerChips();
    });
    loraEditorTriggerChipsEl.append(chip);
  });
}

function renderLoraEditor() {
  const phase = state.loraEditorPhase;
  const uploading = phase === "uploading";
  const analyzing = phase === "analyzing";
  const saving = phase === "saving";
  const editing = phase === "editing";
  loraEditorTitleEl.textContent = state.loraEditorMode === "edit" ? "EDIT LORA" : "ADD LORA";
  loraEditorFileEl.textContent = state.loraEditorSourceFilename || "";
  loraEditorProgressEl.classList.toggle("hidden", editing);
  loraEditorProgressEl.classList.toggle("uploading", uploading);
  loraEditorProgressEl.classList.toggle("analyzing", analyzing);
  loraEditorProgressEl.classList.toggle("saving", saving);
  if (uploading) {
    loraEditorProgressSpinnerEl.textContent = `Uploading... ${Math.round(state.loraEditorUploadProgress || 0)}%`;
  } else if (analyzing) {
    loraEditorProgressSpinnerEl.textContent = "Analyzing metadata...";
  } else if (saving) {
    loraEditorProgressSpinnerEl.textContent = "Saving LoRA...";
  } else {
    loraEditorProgressSpinnerEl.textContent = "";
  }
  loraEditorBodyEl.classList.toggle("hidden", uploading || analyzing);
  loraEditorNameEl.value = state.loraEditorDisplayName || "";
  if (state.loraEditorPreviewUrl) {
    loraEditorThumbnailPreviewEl.src = state.loraEditorPreviewUrl;
    loraEditorThumbnailPreviewEl.classList.remove("empty");
  } else {
    loraEditorThumbnailPreviewEl.removeAttribute("src");
    loraEditorThumbnailPreviewEl.classList.add("empty");
  }
  renderLoraEditorTriggerChips();
  loraEditorNameEl.disabled = !editing;
  loraEditorThumbnailButtonEl.disabled = !editing;
  loraEditorThumbnailInputEl.disabled = !editing;
  loraEditorTriggerInputEl.disabled = !editing;
  loraEditorCloseEl.disabled = saving;
  loraEditorCloseEl.textContent = uploading || analyzing ? "Cancel Upload" : "Cancel";
  loraEditorSaveEl.disabled = !editing || state.loraEditorBusy;
  loraEditorSaveEl.textContent = saving ? "Saving..." : "Save LoRA";
}

function setLoraEditorOpen(open) {
  const next = Boolean(open);
  state.loraEditorOpen = next;
  loraEditorModalEl.classList.toggle("hidden", !next);
  loraEditorModalEl.setAttribute("aria-hidden", String(!next));
  if (next) {
    renderLoraEditor();
    if (loraEditorIsEditing()) {
      loraEditorNameEl.focus();
    } else {
      loraEditorCloseEl.focus();
    }
  }
}

function closeLoraEditor() {
  cleanupLoraEditorPreviewObjectUrl();
  state.loraUploadBusy = false;
  state.loraUploadRequest = null;
  state.loraEditorOpen = false;
  state.loraEditorPhase = "editing";
  state.loraEditorUploadProgress = 0;
  state.loraEditorMode = "create";
  state.loraEditorDraftId = null;
  state.loraEditorLoraId = null;
  state.loraEditorSourceFilename = "";
  state.loraEditorDisplayName = "";
  state.loraEditorTriggerWords = [];
  state.loraEditorDetectedTriggerWords = [];
  state.loraEditorMetadataSummary = {};
  state.loraEditorPreviewUrl = "";
  state.loraEditorPreviewBlob = null;
  state.loraEditorBusy = false;
  loraEditorModalEl.classList.add("hidden");
  loraEditorModalEl.setAttribute("aria-hidden", "true");
}

function openLoraEditorFromDraft(draft) {
  setLoraEditorPreview("");
  state.loraUploadRequest = null;
  state.loraEditorPhase = "editing";
  state.loraEditorUploadProgress = 100;
  state.loraEditorMode = "create";
  state.loraEditorDraftId = String(draft?.draft_id || "").trim();
  state.loraEditorLoraId = null;
  state.loraEditorSourceFilename = String(draft?.source_filename || "").trim();
  state.loraEditorDisplayName = String(draft?.display_name || "").trim();
  state.loraEditorDetectedTriggerWords = Array.isArray(draft?.detected_trigger_words) ? [...draft.detected_trigger_words] : [];
  state.loraEditorTriggerWords = [...state.loraEditorDetectedTriggerWords];
  state.loraEditorMetadataSummary = draft?.metadata_summary || {};
  state.loraEditorBusy = false;
  setLoraEditorOpen(true);
}

function openLoraEditorForItem(item) {
  setLoraEditorPreview(buildLoraPreviewUrl(item));
  state.loraUploadRequest = null;
  state.loraEditorPhase = "editing";
  state.loraEditorUploadProgress = 0;
  state.loraEditorMode = "edit";
  state.loraEditorDraftId = null;
  state.loraEditorLoraId = item.id;
  state.loraEditorSourceFilename = String(item?.source_filename || item?.filename || "").trim();
  state.loraEditorDisplayName = String(item?.display_name || item?.id || "").trim();
  state.loraEditorDetectedTriggerWords = Array.isArray(item?.detected_trigger_words) ? [...item.detected_trigger_words] : [];
  state.loraEditorTriggerWords = Array.isArray(item?.trigger_words) ? [...item.trigger_words] : [];
  state.loraEditorMetadataSummary = item?.metadata_summary || {};
  state.loraEditorBusy = false;
  setLoraEditorOpen(true);
}

function addLoraEditorTriggerWord(rawValue) {
  const value = String(rawValue || "").trim().replace(/\s+/g, " ");
  if (!value) return false;
  const lowered = value.toLowerCase();
  if (state.loraEditorTriggerWords.some((item) => item.toLowerCase() === lowered)) {
    return false;
  }
  state.loraEditorTriggerWords = [...state.loraEditorTriggerWords, value];
  renderLoraEditorTriggerChips();
  return true;
}

function mergeLoraEditorDetectedTriggerWords(values) {
  let added = 0;
  (Array.isArray(values) ? values : []).forEach((value) => {
    if (addLoraEditorTriggerWord(value)) {
      added += 1;
    }
  });
  return added;
}

function uploadLoraDraftWithProgress(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.loraUploadRequest = xhr;
    const body = new FormData();
    body.append("file", file, file.name);

    xhr.open("POST", "/lora-drafts", true);
    for (const [key, value] of buildClientHeaders()) {
      xhr.setRequestHeader(key, value);
    }

    xhr.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      state.loraEditorPhase = "uploading";
      state.loraEditorUploadProgress = Math.max(0, Math.min(100, (event.loaded / event.total) * 100));
      renderLoraEditor();
    });

    xhr.upload.addEventListener("load", () => {
      state.loraEditorPhase = "analyzing";
      state.loraEditorUploadProgress = 100;
      renderLoraEditor();
    });

    xhr.addEventListener("load", () => {
      let payload = null;
      try {
        payload = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch (_) {
        payload = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
        return;
      }
      reject(new Error(formatApiError(payload, "Failed to stage LoRA.")));
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Failed to stage LoRA."));
    });

    xhr.addEventListener("abort", () => {
      const error = new Error("LoRA upload cancelled.");
      error.name = "AbortError";
      reject(error);
    });

    xhr.send(body);
  });
}

async function createCenteredSquareThumbnail(file) {
  const objectUrl = URL.createObjectURL(file);
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error("Failed to load thumbnail image."));
      element.src = objectUrl;
    });
    const side = Math.min(image.naturalWidth || image.width, image.naturalHeight || image.height);
    const sx = Math.max(0, Math.floor(((image.naturalWidth || image.width) - side) / 2));
    const sy = Math.max(0, Math.floor(((image.naturalHeight || image.height) - side) / 2));
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 1024;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Canvas is unavailable in this browser.");
    }
    context.drawImage(image, sx, sy, side, side, 0, 0, 1024, 1024);
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((result) => {
        if (result) {
          resolve(result);
        } else {
          reject(new Error("Failed to generate thumbnail."));
        }
      }, "image/png");
    });
    return { blob, url: URL.createObjectURL(blob) };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function createLoraAddTile() {
  const tile = document.createElement("button");
  tile.type = "button";
  tile.className = "lora-card lora-add-tile";
  tile.title = "Add a LoRA by browsing your local files and giving it a name and a thumbnail.";
  const plus = document.createElement("span");
  plus.className = "lora-add-plus";
  plus.textContent = "+";
  const body = document.createElement("div");
  body.className = "lora-add-body";
  const label = document.createElement("div");
  label.className = "lora-add-label";
  label.textContent = "ADD LORA";
  const hint = document.createElement("div");
  hint.className = "lora-add-hint";
  hint.textContent = "Browse local files, then choose a name, triggers, and thumbnail.";
  body.append(label, hint);
  tile.append(plus, body);
  tile.addEventListener("click", () => {
    if (!Boolean(state.loraCapabilities?.supported) || state.loraUploadBusy) return;
    loraUploadInputEl.click();
  });
  return tile;
}

function getLoraMasonryColumnCount() {
  return window.innerWidth <= LORA_MASONRY_SINGLE_COLUMN_BREAKPOINT_PX ? 1 : 2;
}

function getLoraMasonryOrder(element) {
  return Number.parseInt(String(element?.dataset?.masonryOrder || "0"), 10) || 0;
}

function applyLoraPreviewAspectRatio(preview) {
  if (!(preview instanceof HTMLImageElement)) return;
  const naturalWidth = Number(preview.naturalWidth || 0);
  const naturalHeight = Number(preview.naturalHeight || 0);
  preview.style.aspectRatio = naturalWidth > 0 && naturalHeight > 0
    ? `${naturalWidth} / ${naturalHeight}`
    : "1 / 1";
}

function createLoraMasonryColumns(columnCount) {
  loraListEl.innerHTML = "";
  loraListEl.style.setProperty("--lora-masonry-columns", String(columnCount));
  const columns = [];
  for (let index = 0; index < columnCount; index += 1) {
    const column = document.createElement("div");
    column.className = "lora-list-column";
    column.dataset.columnIndex = String(index);
    columns.push(column);
  }
  loraListEl.append(...columns);
  return columns;
}

function appendToShortestLoraMasonryColumn(item, columns, heights) {
  if (columns.length === 0) return;
  let targetIndex = 0;
  for (let index = 1; index < heights.length; index += 1) {
    if (heights[index] < heights[targetIndex]) {
      targetIndex = index;
    }
  }
  columns[targetIndex].append(item);
  heights[targetIndex] = columns[targetIndex].scrollHeight;
}

function placeLoraMasonryItems(items) {
  const columnCount = getLoraMasonryColumnCount();
  const columns = createLoraMasonryColumns(columnCount);
  const heights = new Array(columnCount).fill(0);
  items.forEach((item) => appendToShortestLoraMasonryColumn(item, columns, heights));
}

function relayoutLoraLibraryMasonry() {
  const items = Array.from(loraListEl.querySelectorAll(".lora-card"))
    .sort((left, right) => getLoraMasonryOrder(left) - getLoraMasonryOrder(right));
  if (items.length === 0) {
    loraListEl.innerHTML = "";
    loraListEl.style.setProperty("--lora-masonry-columns", String(getLoraMasonryColumnCount()));
    return;
  }
  placeLoraMasonryItems(items);
}

function scheduleLoraLibraryMasonryRelayout() {
  if (state.loraLibraryMasonryFrame !== null) {
    window.cancelAnimationFrame(state.loraLibraryMasonryFrame);
  }
  state.loraLibraryMasonryFrame = window.requestAnimationFrame(() => {
    state.loraLibraryMasonryFrame = null;
    relayoutLoraLibraryMasonry();
  });
}

function createLoraLibraryCard(item, order) {
  const displayName = String(item?.display_name || item?.id || "").trim() || "Unnamed LoRA";
  const sourceFilename = String(item?.source_filename || item?.filename || "").trim();
  const cardTitle = sourceFilename && sourceFilename !== displayName
    ? `${displayName}\n${sourceFilename}`
    : displayName;
  const pendingSelection = state.loraPendingSelections.find((selection) => selection.id === item.id) || null;
  const appliedSelection = state.loraAppliedSelections.find((selection) => selection.id === item.id) || null;
  const isTouchFocus = state.loraTouchFocusId === item.id;
  const card = document.createElement("article");
  card.className = "lora-card";
  card.classList.toggle("pending", Boolean(pendingSelection));
  card.classList.toggle("applied", Boolean(appliedSelection));
  card.classList.toggle("touch-focused", isTouchFocus);
  card.dataset.loraId = item.id;
  card.dataset.masonryOrder = String(order);
  card.title = cardTitle;

  const preview = document.createElement("img");
  preview.className = "lora-card-preview";
  preview.alt = `${displayName} preview`;
  preview.title = cardTitle;
  preview.loading = "lazy";
  preview.decoding = "async";
  preview.style.aspectRatio = "1 / 1";
  const onPreviewSettled = () => {
    applyLoraPreviewAspectRatio(preview);
    scheduleLoraLibraryMasonryRelayout();
  };
  preview.addEventListener("load", onPreviewSettled, { once: true });
  preview.addEventListener("error", onPreviewSettled, { once: true });
  preview.src = buildLoraPreviewUrl(item);
  if (preview.complete) {
    onPreviewSettled();
  }

  const overlay = document.createElement("div");
  overlay.className = "lora-card-overlay";

  const topRow = document.createElement("div");
  topRow.className = "lora-card-top-row";
  const name = document.createElement("div");
  name.className = "lora-card-name";
  name.textContent = displayName;
  name.title = cardTitle;
  topRow.append(name);

  let actions = null;
  if (!pendingSelection) {
    actions = document.createElement("div");
    actions.className = "lora-card-actions";

    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.className = "lora-card-icon-button";
    editButton.title = "Edit LoRA";
    editButton.setAttribute("aria-label", "Edit LoRA");
    editButton.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h4l10-10-4-4L4 16v4"></path><path d="M13 7l4 4"></path></svg>';
    editButton.addEventListener("click", (event) => {
      event.stopPropagation();
      openLoraEditorForItem(item);
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "lora-card-icon-button danger";
    deleteButton.title = "Delete LoRA";
    deleteButton.setAttribute("aria-label", "Delete LoRA");
    deleteButton.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"></path><path d="M9 7V5h6v2"></path><path d="M7 7l1 12h8l1-12"></path><path d="M10 10v6"></path><path d="M14 10v6"></path></svg>';
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      showConfirmModal(
        `Delete LoRA \"${item.display_name || item.id}\"? This cannot be undone.`,
        async () => {
          await deleteLoraById(item.id);
        },
        "Delete",
        "Cancel"
      );
    });

    actions.append(editButton, deleteButton);
  }

  overlay.append(topRow);
  const centerRow = document.createElement("div");
  centerRow.className = "lora-card-center-row";
  centerRow.append(createLoraToggleButton(item, Boolean(pendingSelection)));
  overlay.append(centerRow);
  if (pendingSelection) {
    const sliderRow = document.createElement("div");
    sliderRow.className = "lora-slider-row";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(LORA_STRENGTH_PRESETS.length - 1);
    slider.step = "1";
    slider.value = String(loraStrengthPresetIndex(pendingSelection.weight));
    slider.addEventListener("click", (event) => event.stopPropagation());
    slider.addEventListener("input", (event) => {
      event.stopPropagation();
      const weight = setPendingLoraWeight(item.id, loraStrengthPresetAt(slider.value).value, { skipRender: true });
      value.textContent = formatLoraPresetLabel(weight);
    });
    slider.addEventListener("change", (event) => {
      event.stopPropagation();
      renderLoraLibrary();
    });
    const value = document.createElement("div");
    value.className = "lora-slider-value";
    value.textContent = formatLoraPresetLabel(pendingSelection.weight);
    sliderRow.append(slider, value);
    overlay.append(sliderRow);
  } else if (actions) {
    const bottomRow = document.createElement("div");
    bottomRow.className = "lora-card-bottom-row";
    bottomRow.append(actions);
    overlay.append(bottomRow);
  }

  card.append(preview, overlay);
  if (isTouchLikeLoraUi()) {
    card.addEventListener("click", () => {
      state.loraTouchFocusId = state.loraTouchFocusId === item.id ? null : item.id;
      renderLoraLibrary();
    });
  }
  return card;
}

function createLoraToggleButton(item, isActive) {
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "lora-toggle-button";
  toggle.classList.toggle("active", Boolean(isActive));
  toggle.setAttribute("aria-pressed", String(Boolean(isActive)));
  toggle.title = isActive ? "Disable LoRA" : "Enable LoRA";

  const track = document.createElement("span");
  track.className = "lora-toggle-track";

  const indicator = document.createElement("span");
  indicator.className = "lora-toggle-indicator";
  indicator.textContent = isActive ? "✓" : "✕";

  const thumb = document.createElement("span");
  thumb.className = "lora-toggle-thumb";

  track.append(indicator, thumb);
  toggle.append(track);
  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    togglePendingLoraSelection(item.id);
  });
  return toggle;
}

function setPendingLoraWeight(loraId, rawValue, options = {}) {
  const target = String(loraId || "").trim();
  if (!target) return DEFAULT_LORA_WEIGHT;
  const normalized = normalizeLoraWeight(rawValue);
  state.loraPendingSelections = state.loraPendingSelections.map((item) =>
    item.id === target ? { ...item, weight: normalized } : item
  );
  if (!options.skipRender) {
    renderLoraLibrary();
  }
  return normalized;
}

function renderLoraLibrary() {
  const filterValue = String(state.loraFilter || "").trim().toLowerCase();
  const filteredItems = state.loraLibrary.filter((item) => {
    if (!filterValue) return true;
    const haystack = `${item.display_name || ""} ${item.source_filename || ""}`.toLowerCase();
    return haystack.includes(filterValue);
  });
  const items = [];
  const addTile = createLoraAddTile();
  addTile.dataset.masonryOrder = "0";
  items.push(addTile);
  if (filteredItems.length === 0) {
    loraDrawerEmptyEl.classList.remove("hidden");
    loraDrawerEmptyEl.textContent =
      state.loraLibrary.length === 0 ? "No LoRAs installed yet." : "No LoRAs match the current filter.";
  } else {
    loraDrawerEmptyEl.classList.add("hidden");
  }
  filteredItems.forEach((item, index) => {
    items.push(createLoraLibraryCard(item, index + 1));
  });
  placeLoraMasonryItems(items);
  scheduleLoraLibraryMasonryRelayout();
}

function syncAppliedLorasWithLibrary() {
  if (!Boolean(state.loraCapabilities?.supported)) {
    const hadApplied = state.loraAppliedSelections.length > 0;
    state.loraAppliedSelections = [];
    state.loraPendingSelections = [];
    if (hadApplied) {
      persistAppliedLorasToSession();
    }
    return;
  }
  if (state.loraTouchFocusId && !state.loraLibrary.some((item) => item.id === state.loraTouchFocusId)) {
    state.loraTouchFocusId = null;
  }
  const sanitizedApplied = sanitizeLoraSelectionsAgainstLibrary(state.loraAppliedSelections);
  const sanitizedPending = sanitizeLoraSelectionsAgainstLibrary(state.loraPendingSelections);
  const appliedChanged = !loraSelectionsEqual(sanitizedApplied, state.loraAppliedSelections);
  state.loraAppliedSelections = sanitizedApplied;
  state.loraPendingSelections = sanitizedPending;
  if (appliedChanged) {
    persistAppliedLorasToSession();
  }
}

function buildLoraLibrarySignature(items, capabilities) {
  const normalizedItems = (Array.isArray(items) ? items : []).map((item) => ({
    id: String(item?.id || ""),
    display_name: String(item?.display_name || ""),
    source_filename: String(item?.source_filename || ""),
    preview_filename: String(item?.preview_filename || ""),
    preview_cache_key: String(item?.preview_cache_key || ""),
    preview_is_custom: Boolean(item?.preview_is_custom),
    trigger_words: Array.isArray(item?.trigger_words) ? item.trigger_words.map((value) => String(value || "")) : [],
    detected_trigger_words: Array.isArray(item?.detected_trigger_words)
      ? item.detected_trigger_words.map((value) => String(value || ""))
      : [],
    created_at: String(item?.created_at || ""),
    updated_at: String(item?.updated_at || ""),
    file_size_bytes: Number(item?.file_size_bytes || 0),
  }));
  const normalizedCapabilities = {
    supported: Boolean(capabilities?.supported),
    active_pack: String(capabilities?.active_pack || ""),
    max_active: Number(capabilities?.max_active || 0),
    min_weight: Number(capabilities?.min_weight || 0),
    max_weight: Number(capabilities?.max_weight || 0),
    default_weight: Number(capabilities?.default_weight || 0),
  };
  return JSON.stringify({
    items: normalizedItems,
    capabilities: normalizedCapabilities,
  });
}

async function loadLoraLibrary(options = {}) {
  const requestId = state.loraLibraryLoadRequestSeq + 1;
  state.loraLibraryLoadRequestSeq = requestId;
  const query = new URLSearchParams({ _: String(Date.now()) });
  const response = await apiFetch(`/loras?${query.toString()}`, { cache: "no-store" });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(formatApiError(payload, "Failed to load LoRAs."));
  }
  if (requestId !== state.loraLibraryLoadRequestSeq) {
    return;
  }
  const nextLibrary = Array.isArray(payload?.items) ? payload.items : [];
  const nextCapabilities = payload?.capabilities || state.loraCapabilities;
  const nextSignature = buildLoraLibrarySignature(nextLibrary, nextCapabilities);
  if (nextSignature === state.loraLibrarySignature) {
    return;
  }
  state.loraLibrary = nextLibrary;
  state.loraCapabilities = nextCapabilities;
  state.loraLibrarySignature = nextSignature;
  syncAppliedLorasWithLibrary();
  updateLoraCapabilityUi();
  renderLoraLibrary();
  if (options.refreshSummary !== false) {
    updateSettingsSummary();
  }
}

function stopLoraLibraryPolling() {
  if (state.loraLibraryPollTimer === null) return;
  window.clearTimeout(state.loraLibraryPollTimer);
  state.loraLibraryPollTimer = null;
}

function scheduleLoraLibraryPoll() {
  if (!state.loraDrawerOpen || document.hidden) {
    stopLoraLibraryPolling();
    return;
  }
  if (state.loraLibraryPollTimer !== null) {
    return;
  }
  state.loraLibraryPollTimer = window.setTimeout(async () => {
    state.loraLibraryPollTimer = null;
    try {
      await loadLoraLibrary({ refreshSummary: false, silent: true });
    } catch (_) {
    } finally {
      if (state.loraDrawerOpen && !document.hidden) {
        scheduleLoraLibraryPoll();
      }
    }
  }, LORA_LIBRARY_POLL_INTERVAL_MS);
}

function stopLoraLibraryEventStream() {
  if (state.loraLibraryEventReconnectTimer !== null) {
    window.clearTimeout(state.loraLibraryEventReconnectTimer);
    state.loraLibraryEventReconnectTimer = null;
  }
  if (state.loraLibraryEventSource === null) {
    return;
  }
  const source = state.loraLibraryEventSource;
  state.loraLibraryEventSource = null;
  source.onmessage = null;
  source.onerror = null;
  try {
    source.close();
  } catch (_) {
  }
}

function scheduleLoraLibraryEventReconnect() {
  if (state.loraLibraryEventSource !== null || state.loraLibraryEventReconnectTimer !== null) {
    return;
  }
  state.loraLibraryEventReconnectTimer = window.setTimeout(() => {
    state.loraLibraryEventReconnectTimer = null;
    startLoraLibraryEventStream();
  }, 3000);
}

function startLoraLibraryEventStream() {
  if (typeof window.EventSource !== "function") {
    return;
  }
  if (state.loraLibraryEventReconnectTimer !== null) {
    window.clearTimeout(state.loraLibraryEventReconnectTimer);
    state.loraLibraryEventReconnectTimer = null;
  }
  if (state.loraLibraryEventSource !== null) {
    return;
  }
  const source = new window.EventSource("/loras/events");
  state.loraLibraryEventSource = source;
  source.onmessage = () => {
    loadLoraLibrary({ refreshSummary: false, silent: true }).catch(() => {
    });
  };
  source.onerror = () => {
    if (state.loraLibraryEventSource !== source) {
      return;
    }
    state.loraLibraryEventSource = null;
    source.onmessage = null;
    source.onerror = null;
    try {
      source.close();
    } catch (_) {
    }
    scheduleLoraLibraryEventReconnect();
  };
}

function togglePendingLoraSelection(loraId) {
  const target = String(loraId || "").trim();
  if (!target) return false;
  const existingIndex = state.loraPendingSelections.findIndex((item) => item.id === target);
  if (existingIndex >= 0) {
    state.loraPendingSelections.splice(existingIndex, 1);
    if (state.loraTouchFocusId === target) {
      state.loraTouchFocusId = null;
    }
    renderLoraLibrary();
    return true;
  }
  if (pendingLoraCount() >= MAX_ACTIVE_LORAS) {
    setStatus(`You can enable up to ${MAX_ACTIVE_LORAS} LoRAs at once.`, true);
    return false;
  }
  if (state.loraTouchFocusId === target) {
    state.loraTouchFocusId = null;
  }
  state.loraPendingSelections.push({ id: target, weight: DEFAULT_LORA_WEIGHT });
  renderLoraLibrary();
  return true;
}

function applyPendingLoras(options = {}) {
  state.loraAppliedSelections = cloneLoraSelections(state.loraPendingSelections);
  persistAppliedLorasToSession();
  updateSettingsSummary();
  renderLoraLibrary();
  const count = appliedLoraCount();
  if (options.closeDrawer) {
    setLoraDrawerOpen(false);
  }
  if (!options.silent) {
    setStatus(count > 0 ? `Applied ${count} LoRA${count === 1 ? "" : "s"}.` : "Cleared applied LoRAs.");
  }
}

async function deleteLoraById(loraId) {
  const target = String(loraId || "").trim();
  if (!target) return false;
  const response = await apiFetch(`/loras/${encodeURIComponent(target)}`, { method: "DELETE" });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(formatApiError(payload, "Failed to delete LoRA."));
  }
  state.loraLibrary = state.loraLibrary.filter((item) => item.id !== target);
  state.loraPendingSelections = state.loraPendingSelections.filter((item) => item.id !== target);
  const previousAppliedCount = appliedLoraCount();
  state.loraAppliedSelections = state.loraAppliedSelections.filter((item) => item.id !== target);
  let queueChanged = false;
  state.queue = state.queue.map((job) => {
    const nextLoras = cloneLoraSelections(job.loras).filter((item) => item.id !== target);
    if (nextLoras.length !== cloneLoraSelections(job.loras).length) {
      queueChanged = true;
      return { ...job, loras: nextLoras };
    }
    return job;
  });
  if (state.activeJob && !state.activeJob.remoteInFlight) {
    const nextLoras = cloneLoraSelections(state.activeJob.loras).filter((item) => item.id !== target);
    if (nextLoras.length !== cloneLoraSelections(state.activeJob.loras).length) {
      state.activeJob = { ...state.activeJob, loras: nextLoras };
      queueChanged = true;
    }
  }
  if (queueChanged) {
    persistClientQueueState();
  }
  if (state.loraTouchFocusId === target) {
    state.loraTouchFocusId = null;
  }
  if (appliedLoraCount() !== previousAppliedCount) {
    persistAppliedLorasToSession();
    updateSettingsSummary();
  }
  renderLoraLibrary();
  setStatus(
    payload?.deletion_state === "deferred"
      ? `Deleted LoRA ${target}. Cleanup will finish after the active generation.`
      : `Deleted LoRA ${target}.`
  );
  return true;
}

async function uploadSelectedLoraFile(file) {
  if (!(file instanceof File)) return false;
  state.loraUploadBusy = true;
  state.loraEditorBusy = true;
  state.loraEditorPhase = "uploading";
  state.loraEditorUploadProgress = 0;
  state.loraEditorMode = "create";
  state.loraEditorDraftId = null;
  state.loraEditorLoraId = null;
  state.loraEditorSourceFilename = String(file.name || "").trim();
  state.loraEditorDisplayName = formatInitialLoraDisplayName(file.name);
  state.loraEditorTriggerWords = [];
  state.loraEditorDetectedTriggerWords = [];
  state.loraEditorMetadataSummary = {};
  setLoraEditorPreview("");
  setLoraEditorOpen(true);
  updateLoraCapabilityUi();
  try {
    const payload = await uploadLoraDraftWithProgress(file);
    openLoraEditorFromDraft(payload?.draft || {});
    setLoraDrawerOpen(true);
    setStatus(`Loaded ${file.name}. Complete the form to save it.`);
    return true;
  } catch (error) {
    if (error?.name === "AbortError") {
      return false;
    }
    closeLoraEditor();
    throw error;
  } finally {
    state.loraUploadBusy = false;
    state.loraUploadRequest = null;
    state.loraEditorBusy = false;
    updateLoraCapabilityUi();
  }
}

async function saveLoraEditor() {
  const displayName = String(loraEditorNameEl.value || "").trim();
  if (!displayName) {
    setStatus("LoRA name is required.", true);
    return false;
  }
  state.loraEditorBusy = true;
  state.loraSaveBusy = true;
  state.loraEditorPhase = "saving";
  updateLoraCapabilityUi();
  renderLoraEditor();
  const body = new FormData();
  body.append("display_name", displayName);
  body.append("trigger_words", JSON.stringify(state.loraEditorTriggerWords));
  if (state.loraEditorPreviewBlob) {
    body.append("thumbnail", state.loraEditorPreviewBlob, "thumbnail.png");
  }
  try {
    const mode = state.loraEditorMode;
    let response;
    if (mode === "edit" && state.loraEditorLoraId) {
      response = await apiFetch(`/loras/${encodeURIComponent(state.loraEditorLoraId)}`, {
        method: "PATCH",
        body,
      });
    } else {
      body.append("draft_id", String(state.loraEditorDraftId || ""));
      response = await apiFetch("/loras", {
        method: "POST",
        body,
      });
    }
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(payload, mode === "edit" ? "Failed to update LoRA." : "Failed to save LoRA."));
    }
    await loadLoraLibrary();
    closeLoraEditor();
    setLoraDrawerOpen(true);
    setStatus(
      mode === "edit"
        ? `Updated ${payload?.item?.display_name || displayName}.`
        : `Saved ${payload?.item?.display_name || displayName}.`
    );
    return true;
  } catch (error) {
    setStatus(String(error?.message || error), true);
    return false;
  } finally {
    state.loraEditorBusy = false;
    state.loraSaveBusy = false;
    if (state.loraEditorOpen) {
      state.loraEditorPhase = "editing";
    }
    updateLoraCapabilityUi();
    if (state.loraEditorOpen) {
      renderLoraEditor();
    }
  }
}

function cancelLoraEditor() {
  if (state.loraEditorPhase === "saving") return;
  const activeUpload = state.loraUploadRequest;
  const wasUploading = loraEditorIsUploading() && activeUpload;
  if (activeUpload) {
    state.loraUploadRequest = null;
    try {
      activeUpload.abort();
    } catch (_) {
    }
  }
  closeLoraEditor();
  updateLoraCapabilityUi();
  if (wasUploading) {
    setStatus("LoRA upload cancelled.");
  }
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

function isFavoriteItem(item) {
  return Boolean(Number(item?.favorite) || 0);
}

function buildGalleryQueryState() {
  return {
    prompt: String(filterInputEl.value || "").trim(),
    color: state.activeColorFilter || null,
    favoritesOnly: Boolean(state.favoritesOnly),
  };
}

function galleryQueryStatesEqual(left, right) {
  if (!left || !right) return false;
  return (
    left.prompt === right.prompt &&
    left.color === right.color &&
    Boolean(left.favoritesOnly) === Boolean(right.favoritesOnly)
  );
}

function resolveGalleryRefreshMotion(previousQueryState, nextQueryState, fallback = "default") {
  if (!previousQueryState || galleryQueryStatesEqual(previousQueryState, nextQueryState)) {
    return fallback;
  }
  return "filter-soft";
}

function classifyCornerTone(imageEl) {
  if (!(imageEl instanceof HTMLImageElement)) return "dark";
  if (!imageEl.complete || !imageEl.naturalWidth || !imageEl.naturalHeight) return "dark";
  try {
    const sourceWidth = imageEl.naturalWidth;
    const sourceHeight = imageEl.naturalHeight;
    const sampleWidth = Math.max(16, Math.floor(sourceWidth * 0.18));
    const sampleHeight = Math.max(16, Math.floor(sourceHeight * 0.18));
    const sx = Math.max(0, sourceWidth - sampleWidth);
    const sy = 0;
    const canvas = document.createElement("canvas");
    canvas.width = 24;
    canvas.height = 24;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return "dark";
    context.drawImage(imageEl, sx, sy, sampleWidth, sampleHeight, 0, 0, 24, 24);
    const { data } = context.getImageData(0, 0, 24, 24);
    let luminanceTotal = 0;
    let weightTotal = 0;
    for (let index = 0; index < data.length; index += 4) {
      const alpha = Number(data[index + 3] || 0) / 255;
      if (alpha <= 0.08) continue;
      const red = Number(data[index] || 0);
      const green = Number(data[index + 1] || 0);
      const blue = Number(data[index + 2] || 0);
      const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
      luminanceTotal += luminance * alpha;
      weightTotal += alpha;
    }
    if (weightTotal <= 0) return "dark";
    return luminanceTotal / weightTotal >= 168 ? "light" : "dark";
  } catch (_) {
    return "dark";
  }
}

function getFavoriteToneForFilename(filename) {
  const target = String(filename || "").trim();
  if (!target) return null;
  return state.favoriteToneByFilename.get(target) || null;
}

function setFavoriteToneForFilename(filename, tone) {
  const target = String(filename || "").trim();
  if (!target || (tone !== "light" && tone !== "dark")) return;
  state.favoriteToneByFilename.set(target, tone);
}

function applyFavoriteToneClass(buttonEl, tone) {
  if (!(buttonEl instanceof HTMLElement)) return;
  if (tone !== "light" && tone !== "dark") return;
  buttonEl.classList.toggle("tone-light", tone === "light");
  buttonEl.classList.toggle("tone-dark", tone !== "light");
}

function applyFavoriteButtonTone(buttonEl, imageEl, filename) {
  let tone = getFavoriteToneForFilename(filename);
  if (!tone && imageEl instanceof HTMLImageElement && imageEl.complete && imageEl.naturalWidth > 0) {
    tone = classifyCornerTone(imageEl);
    setFavoriteToneForFilename(filename, tone);
  }
  applyFavoriteToneClass(buttonEl, tone || "dark");
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
    loras: cloneLoraSelections(overrides.loras ?? rawJob.loras),
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
    loras: cloneLoraSelections(job.loras),
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
      .filter((job) => !isActiveJobStatus(job.status))
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

function countVisibleGalleryNodes() {
  let count = 0;
  const nodes = galleryEl.querySelectorAll("[data-gallery-key]");
  for (const node of nodes) {
    if (!node.classList.contains("removing")) {
      count += 1;
    }
  }
  return count;
}

function hasPendingSoftRemovalTiles() {
  for (const batch of state.gallerySoftRemovalBatches.values()) {
    if (batch.tiles.size > 0) {
      return true;
    }
  }
  return false;
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

function resolveGalleryRelayoutDecision(motion, previousCount, nextCount) {
  const safePrevious = Math.max(0, Number(previousCount) || 0);
  const safeNext = Math.max(0, Number(nextCount) || 0);
  if (motion === "none") {
    return { animateLayout: false, removalMode: "immediate", delayEmptyState: false };
  }
  if (motion === "filter-soft") {
    return {
      animateLayout: false,
      removalMode: "batched-soft",
      delayEmptyState: safeNext === 0,
    };
  }
  if (motion === "adaptive") {
    const largestCount = Math.max(safePrevious, safeNext);
    const delta = Math.abs(safePrevious - safeNext);
    const animate = largestCount <= 48 && delta <= 12;
    return {
      animateLayout: animate,
      removalMode: animate ? "legacy" : "immediate",
      delayEmptyState: false,
    };
  }
  return { animateLayout: true, removalMode: "legacy", delayEmptyState: false };
}

function syncGalleryEmptyState(options = {}) {
  const explicitVisibleCount = options.visibleCount;
  const visibleCount =
    explicitVisibleCount === undefined || explicitVisibleCount === null
      ? countVisibleGalleryNodes()
      : Math.max(0, Number(explicitVisibleCount) || 0);
  const shouldDelay = Boolean(options.delayEmptyState) && visibleCount === 0 && hasPendingSoftRemovalTiles();
  emptyStateEl.classList.toggle("hidden", visibleCount > 0 || shouldDelay);
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
  if (!tile) return;
  let shouldSyncEmptyState = false;
  const softBatch = tile?._gallerySoftRemovalBatch || null;
  if (softBatch) {
    softBatch.tiles.delete(tile);
    if (softBatch.timeoutId !== null && softBatch.tiles.size === 0) {
      window.clearTimeout(softBatch.timeoutId);
      softBatch.timeoutId = null;
      state.gallerySoftRemovalBatches.delete(softBatch.revision);
      shouldSyncEmptyState = true;
    }
    tile._gallerySoftRemovalBatch = null;
  }
  const handle = tile?._galleryRemovalHandle;
  if (handle) {
    handle.cancelled = true;
    if (handle.timeoutId !== null) {
      window.clearTimeout(handle.timeoutId);
    }
    tile._galleryRemovalHandle = null;
  }
  tile.classList.remove("removing");
  if (shouldSyncEmptyState) {
    syncGalleryEmptyState();
  }
}

function getGallerySoftRemovalBatch(revision) {
  let batch = state.gallerySoftRemovalBatches.get(revision) || null;
  if (batch) {
    return batch;
  }
  batch = {
    revision,
    tiles: new Set(),
    timeoutId: null,
  };
  state.gallerySoftRemovalBatches.set(revision, batch);
  return batch;
}

function finalizeGallerySoftRemovalBatch(batch) {
  if (!batch) return;
  if (batch.timeoutId !== null) {
    window.clearTimeout(batch.timeoutId);
    batch.timeoutId = null;
  }
  state.gallerySoftRemovalBatches.delete(batch.revision);
  for (const tile of [...batch.tiles]) {
    batch.tiles.delete(tile);
    if (tile._gallerySoftRemovalBatch !== batch) {
      continue;
    }
    tile._gallerySoftRemovalBatch = null;
    if (tile.parentElement === galleryEl) {
      tile.remove();
    }
  }
  syncGalleryEmptyState();
}

function scheduleSoftTileRemoval(tile, revision) {
  if (!tile) return;
  if (tile.classList.contains("removing")) {
    cancelScheduledTileRemoval(tile);
  }
  tile.classList.add("removing");
  const batch = getGallerySoftRemovalBatch(revision);
  batch.tiles.add(tile);
  tile._gallerySoftRemovalBatch = batch;
  if (batch.timeoutId !== null) {
    window.clearTimeout(batch.timeoutId);
  }
  batch.timeoutId = window.setTimeout(() => {
    finalizeGallerySoftRemovalBatch(batch);
  }, GALLERY_SOFT_REMOVAL_DURATION_MS);
}

function scheduleTileRemoval(tile) {
  if (!tile) return;
  if (tile.classList.contains("removing")) {
    cancelScheduledTileRemoval(tile);
  }
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

function removeTileImmediately(tile) {
  if (!tile) return;
  cancelScheduledTileRemoval(tile);
  if (tile.parentElement === galleryEl) {
    tile.remove();
  }
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
  multiSelectToggleEl.classList.toggle("active", active);
  multiSelectToggleEl.setAttribute("aria-label", active ? "Disable multiselection" : "Enable multiselection");
  multiSelectToggleEl.title = active ? "Disable multiselection" : "Enable multiselection";
  multiSelectToggleEl.innerHTML = active
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12"></path><path d="M18 6L6 18"></path></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h10v10H5z"></path><path d="M9 3h10v10"></path><path d="M8 12l2 2 4-4"></path></svg>';
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

function getEffectiveUpscaleScale() {
  return 2;
}

function updateUpscaleControls() {
  state.upscaleMode = "fast";
  state.upscaleScale = 2;
  upscaleSettingsHintEl.textContent = "Upscale uses SeedVR2 direct x2 faithful mode.";
}

function setUpscaleMode(mode) {
  state.upscaleMode = "fast";
  state.upscaleScale = 2;
  updateUpscaleControls();
  updateSettingsSummary();
}

function setUpscaleScale(scale) {
  state.upscaleMode = "fast";
  state.upscaleScale = 2;
  updateUpscaleControls();
  updateSettingsSummary();
}

function updateSettingsSummary() {
  const dimensions = parseResolution(resolutionSelectEl.value);
  const pieces = [
    `Resolution <span class="summary-value">${dimensions.width}x${dimensions.height}</span>`,
    `Enhancer <span class="summary-value">${state.promptEnhance ? "ON" : "OFF"}</span>`,
    `Upscale <span class="summary-value">SEED X2</span>`,
  ];

  if (state.freezeSeed && state.currentSeed !== null) {
    pieces.push(`Seed <span class="summary-value">${state.currentSeed}</span>`);
  }
  if (state.proceduralCreativity > 0) {
    pieces.push(
      `Creative Mode <span class="summary-value">${describeProceduralCreativity(state.proceduralCreativity)}</span>`,
    );
  }
  if (appliedLoraCount() > 0) {
    pieces.push(`LoRAs <span class="summary-value">${appliedLoraCount()}</span>`);
  }

  settingsSummaryEl.innerHTML = pieces
    .map((piece, index) => (index === 0 ? piece : `<span class="summary-sep">|</span> ${piece}`))
    .join(" ");
  updateTopbarOffset();
}

function updateReverseButton() {
  if (state.newestFirst) {
    reverseOrderButtonEl.classList.remove("reversed");
    reverseOrderButtonEl.setAttribute("aria-label", "Sort by newest first");
    reverseOrderButtonEl.title = "Sort by newest first";
    reverseOrderButtonEl.setAttribute("aria-pressed", "false");
    reverseOrderButtonEl.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 6h10"></path><path d="M8 11h7"></path><path d="M8 16h4"></path><path d="M5 5v14"></path><path d="M2.5 16.5L5 19l2.5-2.5"></path></svg>';
  } else {
    reverseOrderButtonEl.classList.add("reversed");
    reverseOrderButtonEl.setAttribute("aria-label", "Sort by oldest first");
    reverseOrderButtonEl.title = "Sort by oldest first";
    reverseOrderButtonEl.setAttribute("aria-pressed", "true");
    reverseOrderButtonEl.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 6h4"></path><path d="M8 11h7"></path><path d="M8 16h10"></path><path d="M5 19V5"></path><path d="M2.5 7.5L5 5l2.5 2.5"></path></svg>';
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

function updateFavoriteFilterButton() {
  const active = state.favoritesOnly;
  galleryFavoriteFilterButtonEl.classList.toggle("active", active);
  galleryFavoriteFilterButtonEl.setAttribute("aria-pressed", String(active));
}

async function setActiveColorFilter(color) {
  const normalized = GALLERY_COLOR_FILTERS.includes(String(color || "").trim().toLowerCase())
    ? String(color || "").trim().toLowerCase()
    : null;
  state.activeColorFilter = state.activeColorFilter === normalized ? null : normalized;
  updateColorSwatches();
  await loadImages();
}

async function toggleFavoriteFilter() {
  state.favoritesOnly = !state.favoritesOnly;
  updateFavoriteFilterButton();
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
  clearViewerActionFx();
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
  const favorite = isFavoriteItem(item);
  viewerUpscaleButtonEl.classList.toggle("hidden", upscaled);
  viewerUpscaleButtonEl.disabled = upscaled;
  viewerFavoriteButtonEl.classList.remove("tone-light", "tone-dark");
  viewerFavoriteButtonEl.classList.toggle("active", favorite);
  viewerFavoriteButtonEl.setAttribute("aria-pressed", String(favorite));
  viewerFavoriteButtonEl.setAttribute("aria-label", favorite ? "Unfavorite image" : "Favorite image");
  viewerFavoriteButtonEl.title = favorite ? "Unfavorite" : "Favorite";
  applyFavoriteButtonTone(viewerFavoriteButtonEl, viewerImageEl, item.filename);
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
    const loras = parseImageLoras(item);
    const loraLabel =
      loras.length > 0
        ? loras.map((entry) => `${entry.name || entry.id} ${formatLoraWeight(entry.weight)}`).join(", ")
        : "";
    viewerMetaEl.classList.toggle("expanded", state.viewerPromptExpanded);
    const parts = [
      `<span class="viewer-meta-timestamp">${escapeHtml(formatTimestamp(timestamp))}</span>`,
      '<span class="viewer-meta-sep">|</span>',
      `<button type="button" class="viewer-meta-prompt${state.viewerPromptExpanded ? " expanded" : ""}" title="${promptTitle}">${escapeHtml(promptDisplay)}</button>`,
      '<span class="viewer-meta-sep">|</span>',
      `<span>${escapeHtml(resolution)}</span>`,
      '<span class="viewer-meta-sep">|</span>',
      `<span>${escapeHtml(pack)}</span>`,
    ];
    if (loraLabel) {
      parts.push('<span class="viewer-meta-sep">|</span>');
      parts.push(`<span>${escapeHtml(loraLabel)}</span>`);
    }
    viewerMetaEl.innerHTML = parts.join(" ");
  }
  viewerDownloadEl.href = buildDownloadUrl(item.filename);
  viewerDownloadEl.setAttribute("download", item.filename);
}

function showViewer(item, index = -1) {
  if (index < 0) {
    index = state.galleryItems.findIndex((candidate) => candidate.filename === item.filename);
  }
  clearViewerActionFx();
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

function applyPromptFromItem(item, options = {}) {
  if (!item) return false;
  promptInputEl.value = String(item.prompt || "");
  updateTopbarOffset();
  if (options.closeViewer) {
    hideViewer();
  }
  promptInputEl.focus();
  setStatus("Loaded prompt into top bar.");
  return true;
}

function getGalleryTileByFilename(filename) {
  const target = String(filename || "").trim();
  if (!target) return null;
  return [...galleryEl.querySelectorAll("[data-gallery-key]")].find(
    (node) => node.dataset.galleryKey === galleryKeyForImage({ filename: target })
  ) || null;
}

function refreshGalleryItemUi(filename) {
  const item = getGalleryImageItem(filename);
  if (!item) return false;
  const tile = getGalleryTileByFilename(filename);
  if (tile) {
    updateImageTile(tile, item);
  }
  if (state.viewerFilename === filename) {
    applyViewerItemMeta(item);
  }
  syncSelectedFilenames();
  updateMultiSelectControls();
  updateGalleryCount(state.galleryItems.length);
  return true;
}

function updateGalleryItemFavoriteState(filename, favorite, options = {}) {
  const target = String(filename || "").trim();
  if (!target) return { updatedItem: null, structuralChange: false };
  const structuralChange = Boolean(options.forceRender) || (state.favoritesOnly && !favorite);
  let updatedItem = null;
  if (structuralChange) {
    state.galleryItems = state.galleryItems.filter((item) => item.filename !== target);
  } else {
    state.galleryItems = state.galleryItems.map((item) => {
      if (item.filename !== target) {
        return item;
      }
      updatedItem = { ...item, favorite: favorite ? 1 : 0 };
      return updatedItem;
    });
  }
  if (structuralChange) {
    renderGallery({ motion: options.motion || "none" });
    syncViewerWithGallery();
    return { updatedItem: null, structuralChange: true };
  }
  refreshGalleryItemUi(target);
  return { updatedItem, structuralChange: false };
}

async function setImageFavorite(filename, favorite) {
  const target = String(filename || "").trim();
  if (!target) return null;
  const previousItems = state.galleryItems;
  const nextFavorite = Boolean(favorite);
  const optimisticMotion = state.favoritesOnly && !nextFavorite ? "filter-soft" : "none";
  const optimisticResult = updateGalleryItemFavoriteState(target, nextFavorite, { motion: optimisticMotion });
  const response = await apiFetch(`/images/${encodeURIComponent(target)}/favorite`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite: nextFavorite }),
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    state.galleryItems = previousItems;
    renderGallery({ motion: "none" });
    syncViewerWithGallery();
    throw new Error(formatApiError(payload, "Failed to update favorite."));
  }
  const confirmedFavorite = Boolean(payload?.favorite);
  if (confirmedFavorite !== nextFavorite) {
    updateGalleryItemFavoriteState(target, confirmedFavorite, {
      motion: optimisticResult.structuralChange ? "filter-soft" : "default",
    });
  }
  setStatus(confirmedFavorite ? `Favorited ${target}.` : `Removed ${target} from favorites.`);
  return payload?.item || null;
}

async function toggleImageFavoriteByFilename(filename) {
  const item = getGalleryImageItem(filename) || (state.viewerFilename === filename ? getActiveViewerItem() : null);
  if (!item) return null;
  return setImageFavorite(item.filename, !isFavoriteItem(item));
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
  applyPromptFromItem(item, { closeViewer: true });
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
  if (enqueueUpscaleFromItem(item)) {
    playViewerActionFx("upscale");
  }
}

function onViewerDelete() {
  const item = getActiveViewerItem();
  if (!item) return;
  showConfirmModal(
    `Delete "${item.filename}"?\nThis cannot be undone.`,
    async () => {
      await deleteImage(item.filename, { viewerFx: true });
    },
    "Delete",
    "Cancel"
  );
}

function onViewerFavoriteToggle() {
  const item = getActiveViewerItem();
  if (!item) return;
  toggleImageFavoriteByFilename(item.filename).catch((error) => setStatus(String(error?.message || error), true));
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
  state.pendingUpscaleFxIds.delete(target);
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
  tile.classList.toggle("upscale-job", job.kind === "upscale");
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
  image.addEventListener("load", () => {
    const filename = tile.dataset.filename || "";
    const favoriteButton = tile.querySelector(".tile-favorite-button");
    applyFavoriteButtonTone(favoriteButton, image, filename);
  });
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
  const cornerActions = document.createElement("div");
  cornerActions.className = "tile-corner-actions";
  const favoriteButton = document.createElement("button");
  favoriteButton.className = "tile-favorite-button";
  favoriteButton.type = "button";
  favoriteButton.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 21.35 10.55 20C5.4 15.24 2 12.09 2 8.25 2 5.3 4.3 3 7.25 3c1.67 0 3.27.77 4.3 1.98A5.71 5.71 0 0 1 15.85 3C18.8 3 21.1 5.3 21.1 8.25c0 3.84-3.4 6.99-8.55 11.76L12 21.35Z"/></svg>';
  favoriteButton.addEventListener("click", (event) => {
    event.stopPropagation();
    const filename = tile.dataset.filename || "";
    toggleImageFavoriteByFilename(filename).catch((error) => setStatus(String(error?.message || error), true));
  });
  cornerActions.append(favoriteButton, selectBadge);

  const meta = document.createElement("div");
  meta.className = "tile-meta";

  const actions = document.createElement("div");
  actions.className = "tile-actions";
  const primaryActions = document.createElement("div");
  primaryActions.className = "tile-primary-actions";

  const download = document.createElement("a");
  download.className = "tile-download";
  download.title = "Download image";
  download.setAttribute("aria-label", "Download image");
  download.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v10"></path><path d="M8 10l4 4 4-4"></path><path d="M5 18h14"></path></svg>';
  download.addEventListener("click", (event) => event.stopPropagation());
  primaryActions.append(download);

  const upscale = document.createElement("button");
  upscale.className = "tile-upscale";
  upscale.type = "button";
  upscale.title = "Upscale image";
  upscale.setAttribute("aria-label", "Upscale image");
  upscale.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 16L16 8"></path><path d="M10 8h6v6"></path><path d="M4 20h6v-2H6v-4H4z"></path><path d="M20 4h-6v2h4v4h2z"></path></svg>';
  upscale.addEventListener("click", (event) => {
    event.stopPropagation();
    const liveItem = getGalleryImageItem(tile.dataset.filename);
    if (liveItem) {
      enqueueUpscaleFromItem(liveItem);
    }
  });
  primaryActions.append(upscale);

  const usePrompt = document.createElement("button");
  usePrompt.className = "tile-use-prompt";
  usePrompt.type = "button";
  usePrompt.title = "Use prompt text";
  usePrompt.setAttribute("aria-label", "Use prompt text");
  usePrompt.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 6h10"></path><path d="M7 10h10"></path><path d="M7 14h6"></path><path d="M5 4h14v16H5z"></path></svg>';
  usePrompt.addEventListener("click", (event) => {
    event.stopPropagation();
    const liveItem = getGalleryImageItem(tile.dataset.filename);
    if (liveItem) {
      applyPromptFromItem(liveItem);
    }
  });
  primaryActions.append(usePrompt);

  const del = document.createElement("button");
  del.className = "tile-delete tile-delete-text";
  del.type = "button";
  del.title = "Delete image";
  del.setAttribute("aria-label", "Delete image");
  del.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"></path><path d="M9 7V5h6v2"></path><path d="M7 7l1 12h8l1-12"></path><path d="M10 10v6"></path><path d="M14 10v6"></path></svg>';
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    const filename = tile.dataset.filename || "";
    showConfirmModal(`Delete "${filename}"?\nThis cannot be undone.`, async () => {
      await deleteImage(filename);
    }, "Delete");
  });

  primaryActions.append(del);
  actions.append(primaryActions);
  overlay.append(meta, actions);
  tile.append(image, badges, cornerActions, overlay);
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
  clearTileActionFx(tile, { onlyKind: "delete" });
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
  const meta = tile.querySelector(".tile-meta");
  if (meta) {
    meta.textContent = formatGalleryTimestamp(timestamp);
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

  const favoriteButton = tile.querySelector(".tile-favorite-button");
  if (favoriteButton instanceof HTMLButtonElement) {
    const favorite = isFavoriteItem(item);
    favoriteButton.classList.toggle("active", favorite);
    favoriteButton.setAttribute("aria-pressed", String(favorite));
    favoriteButton.setAttribute("aria-label", favorite ? `Unfavorite ${item.filename}` : `Favorite ${item.filename}`);
    favoriteButton.title = favorite ? "Unfavorite" : "Favorite";
    applyFavoriteButtonTone(favoriteButton, image, item.filename);
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

function renderGallery(options = {}) {
  const motion = options.motion || "default";
  const renderRevision = ++state.galleryRenderRevision;
  syncSelectedFilenames();
  updateMultiSelectControls();
  updateGalleryCount(state.galleryItems.length);
  const desiredEntries = buildDesiredGalleryEntries();
  const existingMap = getGalleryNodeMap();
  const relayoutDecision = resolveGalleryRelayoutDecision(
    motion,
    countVisibleGalleryNodes(),
    desiredEntries.length
  );
  const hasContent = desiredEntries.length > 0;
  if (!hasContent) {
    for (const tile of galleryEl.querySelectorAll("[data-gallery-key]")) {
      if (relayoutDecision.removalMode === "legacy") {
        scheduleTileRemoval(tile);
      } else if (relayoutDecision.removalMode === "batched-soft") {
        scheduleSoftTileRemoval(tile, renderRevision);
      } else {
        removeTileImmediately(tile);
      }
    }
    if (relayoutDecision.removalMode === "batched-soft") {
      scheduleGalleryRelayout({ animate: false });
    } else {
      galleryEl.style.height = "";
    }
    syncGalleryEmptyState({ visibleCount: 0, delayEmptyState: relayoutDecision.delayEmptyState });
    return;
  }
  for (const entry of desiredEntries) {
    let tile = existingMap.get(entry.key) || null;
    let created = false;
    if (!tile) {
      created = true;
      tile = entry.kind === "pending" ? createPendingTile(entry.value) : createImageTile(entry.value);
    } else if (entry.kind === "pending") {
      updatePendingTile(tile, entry.value);
    } else {
      updateImageTile(tile, entry.value);
    }
    galleryEl.append(tile);
    if (
      created &&
      entry.kind === "pending" &&
      entry.value?.kind === "upscale" &&
      consumePendingUpscaleEntryFx(entry.value.placeholderId)
    ) {
      playTileActionFxOnTile(tile, "pending-upscale-enter", {
        duration: PENDING_UPSCALE_ENTRY_FX_DURATION_MS,
      });
    }
    existingMap.delete(entry.key);
  }

  for (const tile of existingMap.values()) {
    if (relayoutDecision.removalMode === "legacy") {
      scheduleTileRemoval(tile);
    } else if (relayoutDecision.removalMode === "batched-soft") {
      scheduleSoftTileRemoval(tile, renderRevision);
    } else {
      removeTileImmediately(tile);
    }
  }

  scheduleGalleryRelayout({ animate: relayoutDecision.animateLayout });
  syncGalleryEmptyState({ visibleCount: desiredEntries.length, delayEmptyState: relayoutDecision.delayEmptyState });
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

async function loadImages(options = {}) {
  const requestSeq = ++state.galleryLoadRequestSeq;
  const query = new URLSearchParams();
  const nextQueryState = buildGalleryQueryState();
  const renderMotion = resolveGalleryRefreshMotion(
    state.lastGalleryQueryState,
    nextQueryState,
    options.motion || "default"
  );
  query.set("limit", "500");
  query.set("newest_first", "true");

  const filterValue = nextQueryState.prompt;
  if (filterValue) {
    query.set("prompt", filterValue);
  }
  if (nextQueryState.color) {
    query.set("color", nextQueryState.color);
  }
  if (nextQueryState.favoritesOnly) {
    query.set("favorite", "true");
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
  state.lastGalleryQueryState = nextQueryState;
  syncSelectedFilenames();
  renderGallery({ motion: renderMotion });
  syncViewerWithGallery();
  syncGalleryColorCacheStatusLine();
}

async function deleteImage(filename, options = {}) {
  const skipReload = Boolean(options.skipReload);
  const suppressStatus = Boolean(options.suppressStatus);
  const viewerFx = Boolean(options.viewerFx);
  const triggerFx = options.triggerFx !== false;
  const existingItems = state.galleryItems;
  const hadItem = existingItems.some((item) => item.filename === filename);
  if (viewerFx && triggerFx) {
    await playViewerActionFx("delete");
  }
  if (hadItem) {
    if (triggerFx) {
      playTileActionFx(filename, "delete");
    }
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
  requestCompletionNotificationPermission();
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
    loras: Boolean(state.loraCapabilities?.supported)
      ? cloneLoraSelections(state.loraAppliedSelections)
      : [],
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
  state.pendingUpscaleFxIds.add(placeholderId);
  persistClientQueueState();
  renderGallery();
  playTileActionFx(sourceFilename, "upscale-source");
  updateGenerateButtonState();
  const outstanding = totalOutstandingJobs();
    setStatus(`Queued SEED x2 upscale for ${sourceFilename} -> ${targetWidth}x${targetHeight}. Queue ${outstanding}/${state.maxQueuedGenerations}.`);
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
          setStatus(
            `Upscaling ${job.filename} -> ${job.width}x${job.height} `
            + `(SEED X2, seed ${job.seed})...`
          );
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
        if (!isUpscaleJob && Array.isArray(job.loras) && job.loras.length > 0) {
          payloadBody.loras = cloneLoraSelections(job.loras);
        }
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
          setStatus(
            `Upscaled ${source} -> ${payload.filename} `
            + `(SEED X2) in ${payload.duration_ms} ms `
            + `(seed ${payload.seed}).`
          );
        } else if (payload.prompt_enhanced) {
          setStatus(`Prompt enhanced, saved ${payload.filename} in ${payload.duration_ms} ms (seed ${payload.seed}).`);
        } else {
          setStatus(`Saved ${payload.filename} in ${payload.duration_ms} ms (seed ${payload.seed}).`);
        }
        showGenerationCompletionNotification(job, payload);
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
    state.loraPendingSelections = cloneLoraSelections(state.loraAppliedSelections);
    setGalleryColumns(state.galleryColumns);
    updateTopbarOffset();
    updateReverseButton();
    updateColorSwatches();
    updateFavoriteFilterButton();
    applyOrientationButtonState();
    updateFreezeSeedButton();
    updateProceduralLatentControls();
    updatePromptEnhanceButton();
    updateUpscaleControls();
    updateMultiSelectControls();
    try {
      await loadLoraLibrary({ refreshSummary: false });
    } catch (error) {
      state.loraLibrary = [];
      state.loraCapabilities = {
        supported: false,
        max_active: MAX_ACTIVE_LORAS,
        min_weight: MIN_LORA_WEIGHT,
        max_weight: MAX_LORA_WEIGHT,
        default_weight: DEFAULT_LORA_WEIGHT,
      };
      updateLoraCapabilityUi();
      renderLoraLibrary();
      setStatus(String(error?.message || error), true);
    }
    updateSettingsSummary();
    updateViewerNavState();
    updateGenerateButtonState();
    startLoraLibraryEventStream();
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
loraDrawerToggleEl.addEventListener("click", () => {
  if (loraDrawerToggleEl.disabled) return;
  state.loraPendingSelections = cloneLoraSelections(
    loraSelectionsDirty() ? state.loraPendingSelections : state.loraAppliedSelections
  );
  renderLoraLibrary();
  setLoraDrawerOpen(!state.loraDrawerOpen);
});
loraDrawerCloseEl.addEventListener("click", () => {
  applyPendingLoras({ closeDrawer: true });
});
loraDrawerBackdropEl.addEventListener("click", () => {
  setLoraDrawerOpen(false);
});
loraFilterInputEl.addEventListener("input", () => {
  state.loraFilter = String(loraFilterInputEl.value || "");
  renderLoraLibrary();
});
loraUploadInputEl.addEventListener("change", () => {
  const [file] = Array.from(loraUploadInputEl.files || []);
  if (!file) return;
  uploadSelectedLoraFile(file)
    .catch((error) => setStatus(String(error?.message || error), true))
    .finally(() => {
      loraUploadInputEl.value = "";
    });
});
loraEditorCloseEl.addEventListener("click", () => {
  cancelLoraEditor();
});
loraEditorModalEl.addEventListener("click", (event) => {
  if (event.target !== loraEditorModalEl) return;
  event.preventDefault();
});
loraEditorThumbnailButtonEl.addEventListener("click", () => {
  if (state.loraEditorBusy) return;
  loraEditorThumbnailInputEl.click();
});
loraEditorThumbnailInputEl.addEventListener("change", () => {
  const [file] = Array.from(loraEditorThumbnailInputEl.files || []);
  if (!file) return;
  createCenteredSquareThumbnail(file)
    .then(({ blob, url }) => {
      setLoraEditorPreview(url, blob);
      renderLoraEditor();
    })
    .catch((error) => setStatus(String(error?.message || error), true))
    .finally(() => {
      loraEditorThumbnailInputEl.value = "";
    });
});
loraEditorNameEl.addEventListener("input", () => {
  state.loraEditorDisplayName = String(loraEditorNameEl.value || "");
});
loraEditorTriggerInputEl.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== ",") return;
  event.preventDefault();
  const value = String(loraEditorTriggerInputEl.value || "").trim();
  if (addLoraEditorTriggerWord(value)) {
    loraEditorTriggerInputEl.value = "";
  }
});
loraEditorSaveEl.addEventListener("click", () => {
  saveLoraEditor().catch((error) => setStatus(String(error?.message || error), true));
});
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
  if (state.loraDrawerOpen) {
    setLoraDrawerOpen(true);
    scheduleLoraLibraryMasonryRelayout();
  }
  updateTopbarOffset();
  scheduleGalleryRelayout({ animate: true });
});
window.addEventListener("focus", () => {
  loadLoraLibrary({ refreshSummary: false, silent: true }).catch(() => {
  });
  startLoraLibraryEventStream();
  scheduleLoraLibraryPoll();
});
window.addEventListener("pageshow", () => {
  loadLoraLibrary({ refreshSummary: false, silent: true }).catch(() => {
  });
  startLoraLibraryEventStream();
  scheduleLoraLibraryPoll();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopLoraLibraryPolling();
    return;
  }
  loadLoraLibrary({ refreshSummary: false, silent: true }).catch(() => {
  });
  startLoraLibraryEventStream();
  scheduleLoraLibraryPoll();
});
window.addEventListener("beforeunload", () => {
  stopLoraLibraryEventStream();
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
  if (target.closest("#gallery-favorite-filter")) {
    toggleFavoriteFilter().catch((error) => setStatus(String(error?.message || error), true));
    return;
  }
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
viewerDeleteButtonEl.addEventListener("click", onViewerDelete);
viewerUsePromptButtonEl.addEventListener("click", onViewerUsePrompt);
viewerCopyPromptButtonEl.addEventListener("click", () => {
  onViewerCopyPrompt().catch((error) => setStatus(String(error?.message || error), true));
});
viewerUpscaleButtonEl.addEventListener("click", onViewerUpscale);
viewerFavoriteButtonEl.addEventListener("click", onViewerFavoriteToggle);
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
viewerImageEl.addEventListener("load", () => {
  if (viewerModalEl.classList.contains("hidden")) return;
  const toneFilename =
    (state.viewerCompareHolding && state.viewerCompareSourceFilename) || state.viewerFilename || "";
  applyFavoriteButtonTone(viewerFavoriteButtonEl, viewerImageEl, toneFilename);
});
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
  if (event.key === "Escape" && !loraEditorModalEl.classList.contains("hidden")) {
    event.preventDefault();
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
      `Delete "${item.filename}"?\nThis cannot be undone.`,
      async () => {
        await deleteImage(item.filename, { viewerFx: true });
      },
      "Delete",
      "Cancel"
    );
    return;
  }
  if (event.key === "Escape") {
    hideConfirmModal();
    hideViewer();
    setSettingsVisible(false);
    setLoraDrawerOpen(false);
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
