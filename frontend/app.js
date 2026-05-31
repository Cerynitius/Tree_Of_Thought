const STORAGE_KEY = "tot-terminal-ui-v1";
const STORAGE_VERSION = 21;
const DEFAULT_TIMEOUT_SECONDS = 600;
const DEFAULT_DEPTH_PRESET = "medium";
const DEFAULT_POLL_INTERVAL_MS = 1_000;
const BUSY_REFRESH_INTERVAL_MS = 150;
const RESULT_CANDIDATE_LIMIT = 6;
const RESULT_ANSWER_KEYS = [
  "final_answer",
  "candidate_answer",
  "answer",
  "result",
  "solution",
  "formula",
  "candidate_formula",
  "expression",
  "equation",
  "derived_equation",
  "derived_relation",
  "target_relation",
  "closed_form",
  "steady_state",
  "output",
];
const RESULT_META_EQUATION_PREFIXES = ["step_focus(", "refine(", "route(", "guidance(", "task("];
const RESULT_META_TEXT_PATTERNS = [
  /\bidentify governing relation\b/i,
  /\bchoose one active correction or closure\b/i,
  /\bexpress the target quantity in known variables\b/i,
  /\bisolate the remaining unknown or boundary condition\b/i,
  /\broute-local planning claim\b/i,
  /\brefine only the current subproblem\b/i,
  /\bselect\s+.+\s+route\b/i,
  /\bcurrent child branch\b/i,
];
const NO_CONCRETE_FINAL_ANSWER_TEXT = "No concrete final answer extracted from this branch yet.";
const DEFAULT_PLANNING_MODEL = "qwen3.5-9b-mlx";
const DEFAULT_MODELING_MODEL = "qwen3.5-9b-mlx";
const DEFAULT_REVIEW_MODEL = "qwen3.5-9b-mlx";
const DEFAULT_NON_TERMINAL_EVALUATION_MODEL = "qwen3.5-9b-mlx";
const LEGACY_QWOPUS_MODEL_FAMILY = "qwopus3.5-9b-v3";
const PREVIOUS_DEFAULT_PLANNING_MODEL = "openai/gpt-oss-120b";
const LEGACY_MODELING_DEFAULTS = ["openai/gpt-oss-120b", "qwen3.6-35b-a3b-ud-mlx"];
const LEGACY_REVIEW_DEFAULTS = ["qwen/qwen3-4b-2507", "qwen3.6-35b-a3b-ud-mlx"];
const LEGACY_NON_TERMINAL_EVALUATION_DEFAULTS = ["qwen2.5-0.5b-instruct-mlx", "qwen/qwen3-1.7b"];

const BASE_MODEL_PRESET = {
  planningModel: DEFAULT_PLANNING_MODEL,
  modelingModel: DEFAULT_MODELING_MODEL,
  reviewModel: DEFAULT_REVIEW_MODEL,
  nonTerminalEvaluationModel: DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
};

const DEPTH_PRESET_PROFILES = {
  low: {
    key: "low",
    label: "LOW DEPTH",
    timeoutSeconds: "120",
    stepLabel: "5-step decomposition",
    divergenceLabel: "tight divergence",
    description: "short runtime, live-only, coarse steps, narrow branching",
    scheduler: {
      depth_preset: "low",
      max_reflections: 1,
      max_tree_depth: 5,
      max_frontier_size: 6,
      max_children_per_expansion: 2,
      max_live_children_per_batch: 2,
      use_local_root_proposal: true,
      use_local_root_evaluation: true,
      use_local_child_proposal: true,
      use_local_child_evaluation: true,
      max_frontier_per_diversity_key: 1,
      children_key: "children",
    },
  },
  medium: {
    key: "medium",
    label: "MEDIUM DEPTH",
    timeoutSeconds: "600",
    stepLabel: "8-step decomposition",
    divergenceLabel: "balanced divergence",
    description: "live-only, balanced steps, balanced branching",
    scheduler: {
      depth_preset: "medium",
      max_reflections: 2,
      max_tree_depth: 8,
      max_frontier_size: 16,
      max_children_per_expansion: 3,
      max_live_children_per_batch: 2,
      use_local_root_proposal: true,
      use_local_root_evaluation: true,
      use_local_child_proposal: true,
      use_local_child_evaluation: true,
      max_frontier_per_diversity_key: 4,
      children_key: "children",
    },
  },
  high: {
    key: "high",
    label: "HIGH DEPTH",
    timeoutSeconds: "1200",
    stepLabel: "10-step decomposition",
    divergenceLabel: "broader search, capped child batches",
    description: "long runtime, live-only, finer steps, wider root branching",
    scheduler: {
      depth_preset: "high",
      max_reflections: 3,
      max_tree_depth: 12,
      max_frontier_size: 24,
      max_children_per_expansion: 4,
      max_live_children_per_batch: 2,
      use_local_root_proposal: true,
      use_local_root_evaluation: true,
      use_local_child_proposal: true,
      use_local_child_evaluation: true,
      max_frontier_per_diversity_key: 6,
      children_key: "children",
    },
  },
};

const RECOMMENDED_MODEL_PRESET = {
  ...BASE_MODEL_PRESET,
  ...DEPTH_PRESET_PROFILES[DEFAULT_DEPTH_PRESET],
};

const FALLBACK_DEFAULT_PROBLEM_CONTEXT = {
  task:
    "Use the modeling model to propose the next reasoning step, then score each step for domain consistency and variable grounding.",
  notes: [
    "The frontend polls the live scheduler state and renders it as an ASCII tree.",
    "Node deletion is routed through the backend review model before the subtree is removed.",
  ],
  known_context: {
    objective: "Derive and prune a useful reasoning tree.",
    expected_output: "concise, structured, and domain-valid intermediate steps",
  },
};

const DEFAULT_SCHEDULER_CONFIG = {
  depth_preset: DEFAULT_DEPTH_PRESET,
  max_reflections: 2,
  max_tree_depth: 8,
  max_frontier_size: 16,
  max_children_per_expansion: 3,
  max_live_children_per_batch: 2,
  use_local_root_proposal: true,
  use_local_root_evaluation: true,
  use_local_child_proposal: true,
  use_local_child_evaluation: true,
  max_frontier_per_diversity_key: 4,
  children_key: "children",
};

const uiState = {
  sessionId: "",
  snapshot: null,
  selectedNodeId: null,
  collapsedNodeIds: new Set(),
  treeModel: emptyTreeModel(),
  searchQuery: "",
  searchMatches: [],
  searchCursor: 0,
  pollingEnabled: true,
  pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
  pollTimer: null,
  busyRefreshTimer: null,
  requestInFlight: false,
  revealSelection: false,
  statusMessage: "SYSTEM READY // no session attached",
  statusTone: "idle",
  lastUpdatedAt: null,
  lastActionTitle: "No request yet.",
  lastActionDetail: "Use Create & Run to start a new session or Load Session to reconnect to an existing one.",
  lastActionTone: "idle",
  statusLogEntries: [],
  statusLogSequence: 0,
};

const dom = {
  statusLine: document.getElementById("statusLine"),
  actionFeedback: document.getElementById("actionFeedback"),
  actionFeedbackTitle: document.getElementById("actionFeedbackTitle"),
  actionFeedbackDetail: document.getElementById("actionFeedbackDetail"),
  statusLog: document.getElementById("statusLog"),
  problemPromptInput: document.getElementById("problemPromptInput"),
  sessionIdInput: document.getElementById("sessionIdInput"),
  attachSessionButton: document.getElementById("attachSessionButton"),
  createSessionButton: document.getElementById("createSessionButton"),
  refreshSessionButton: document.getElementById("refreshSessionButton"),
  exportAnswerButton: document.getElementById("exportAnswerButton"),
  dropSessionButton: document.getElementById("dropSessionButton"),
  searchInput: document.getElementById("searchInput"),
  prevMatchButton: document.getElementById("prevMatchButton"),
  nextMatchButton: document.getElementById("nextMatchButton"),
  searchMeta: document.getElementById("searchMeta"),
  pollingToggle: document.getElementById("pollingToggle"),
  pollIntervalInput: document.getElementById("pollIntervalInput"),
  expandAllButton: document.getElementById("expandAllButton"),
  collapseAllButton: document.getElementById("collapseAllButton"),
  deleteReasonInput: document.getElementById("deleteReasonInput"),
  steerPromptInput: document.getElementById("steerPromptInput"),
  deleteNodeButton: document.getElementById("deleteNodeButton"),
  deleteSteerRunButton: document.getElementById("deleteSteerRunButton"),
  applyLowDepthPresetButton: document.getElementById("applyLowDepthPresetButton"),
  applyMediumDepthPresetButton: document.getElementById("applyMediumDepthPresetButton"),
  applyHighDepthPresetButton: document.getElementById("applyHighDepthPresetButton"),
  depthPresetSummary: document.getElementById("depthPresetSummary"),
  baseUrlInput: document.getElementById("baseUrlInput"),
  planningModelInput: document.getElementById("planningModelInput"),
  modelingModelInput: document.getElementById("modelingModelInput"),
  reviewModelInput: document.getElementById("reviewModelInput"),
  nonTerminalEvaluationModelInput: document.getElementById("nonTerminalEvaluationModelInput"),
  timeoutInput: document.getElementById("timeoutInput"),
  problemContextInput: document.getElementById("problemContextInput"),
  schedulerConfigInput: document.getElementById("schedulerConfigInput"),
  treeStats: document.getElementById("treeStats"),
  selectionStats: document.getElementById("selectionStats"),
  resultsPanel: document.getElementById("resultsPanel"),
  resultSummary: document.getElementById("resultSummary"),
  bestResultCard: document.getElementById("bestResultCard"),
  bestResultScore: document.getElementById("bestResultScore"),
  bestResultMeta: document.getElementById("bestResultMeta"),
  bestResultFormula: document.getElementById("bestResultFormula"),
  candidateResults: document.getElementById("candidateResults"),
  treeViewport: document.getElementById("treeViewport"),
  treeLines: document.getElementById("treeLines"),
  detailSummary: document.getElementById("detailSummary"),
  detailBody: document.getElementById("detailBody"),
  frontierList: document.getElementById("frontierList"),
  activityLog: document.getElementById("activityLog"),
};

boot().catch((error) => {
  handleError(error);
});

async function boot() {
  const defaultDrafts = await applyDefaultDrafts();
  restoreDraft();
  wireEvents();
  pushStatusLog(
    "UI ready.",
    "No session attached yet. Use Create & Run or Load Session to begin.",
    "idle",
  );
  render();

  if (defaultDrafts.warning) {
    setStatus("Backend defaults endpoint unavailable. Using built-in drafts.", "warn", { record: false });
    setActionFeedback("Backend defaults unavailable.", defaultDrafts.warning, "warn", { record: false });
    pushStatusLog("Backend defaults unavailable.", defaultDrafts.warning, "warn");
    render();
  }

  if (dom.sessionIdInput.value.trim()) {
    await attachSession({ silent: true });
  }
}

function emptyTreeModel() {
  return {
    root: null,
    nodeById: new Map(),
    ancestorsById: new Map(),
    visibleNodes: [],
    allNodes: [],
    maxDepth: 0,
    activeCount: 0,
    prunedCount: 0,
    leafCount: 0,
  };
}

async function applyDefaultDrafts() {
  const defaults = await loadDefaultDrafts();
  if (!dom.problemContextInput.value.trim()) {
    dom.problemContextInput.value = JSON.stringify(defaults.problem_context, null, 2);
  }
  if (!dom.schedulerConfigInput.value.trim()) {
    dom.schedulerConfigInput.value = JSON.stringify(defaults.scheduler, null, 2);
  }
  return defaults;
}

async function loadDefaultDrafts() {
  const fallback = {
    problem_context: FALLBACK_DEFAULT_PROBLEM_CONTEXT,
    scheduler: DEFAULT_SCHEDULER_CONFIG,
  };
  const buildFallbackResponse = (reason) => ({
    ...fallback,
    warning: `${reason} Open 127.0.0.1:8000 if this page is not serving /api/tot.`,
  });
  try {
    const response = await fetch("/api/tot/defaults");
    if (!response.ok) {
      return buildFallbackResponse(
        `GET /api/tot/defaults returned HTTP ${response.status}. Using built-in defaults.`
      );
    }
    const payload = await response.json();
    return {
      problem_context: isPlainObject(payload.problem_context)
        ? payload.problem_context
        : fallback.problem_context,
      scheduler: isPlainObject(payload.scheduler)
        ? payload.scheduler
        : fallback.scheduler,
      warning: "",
    };
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return buildFallbackResponse(
      `Failed to load /api/tot/defaults: ${reason}. Using built-in defaults.`
    );
  }
}

function readModelInput(element, fallback) {
  if (!(element instanceof HTMLInputElement)) {
    return fallback;
  }
  return sanitizeModelName(element.value, fallback);
}

function writeModelInput(element, value, fallback) {
  if (!(element instanceof HTMLInputElement)) {
    return;
  }
  element.value = sanitizeModelName(value, fallback);
}

function normalizeDepthPreset(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "low" || normalized === "medium" || normalized === "high") {
    return normalized;
  }
  return DEFAULT_DEPTH_PRESET;
}

function getDepthPresetProfile(presetName) {
  return DEPTH_PRESET_PROFILES[normalizeDepthPreset(presetName)] || DEPTH_PRESET_PROFILES[DEFAULT_DEPTH_PRESET];
}

function readJsonObjectOrFallback(rawValue, fallbackValue) {
  try {
    const parsed = parseJsonText(rawValue, "JSON payload", fallbackValue);
    if (isPlainObject(parsed)) {
      return parsed;
    }
  } catch (_error) {
    return { ...fallbackValue };
  }
  return { ...fallbackValue };
}

function describeFallbackPolicy(_profile) {
  return "live-only, no fallback";
}

function inferSelectedDepthPreset() {
  const scheduler = readJsonObjectOrFallback(dom.schedulerConfigInput.value, DEFAULT_SCHEDULER_CONFIG);
  if (typeof scheduler.depth_preset === "string" && scheduler.depth_preset.trim()) {
    return normalizeDepthPreset(scheduler.depth_preset);
  }
  const problemContext = readJsonObjectOrFallback(dom.problemContextInput.value, FALLBACK_DEFAULT_PROBLEM_CONTEXT);
  if (typeof problemContext.reasoning_depth_preset === "string" && problemContext.reasoning_depth_preset.trim()) {
    return normalizeDepthPreset(problemContext.reasoning_depth_preset);
  }
  return DEFAULT_DEPTH_PRESET;
}

function renderDepthPresetControls() {
  const selectedPreset = inferSelectedDepthPreset();
  const profile = getDepthPresetProfile(selectedPreset);
  const buttonMap = {
    low: dom.applyLowDepthPresetButton,
    medium: dom.applyMediumDepthPresetButton,
    high: dom.applyHighDepthPresetButton,
  };

  Object.entries(buttonMap).forEach(([presetKey, button]) => {
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }
    const isActive = presetKey === selectedPreset;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  });

  if (dom.depthPresetSummary) {
    dom.depthPresetSummary.textContent = `${profile.label}: timeout ${profile.timeoutSeconds}s | ${describeFallbackPolicy(profile)} | ${profile.stepLabel} | ${profile.divergenceLabel}.`;
  }
}

function applyDepthPreset(presetName) {
  const profile = getDepthPresetProfile(presetName);

  writeModelInput(
    dom.planningModelInput,
    BASE_MODEL_PRESET.planningModel,
    DEFAULT_PLANNING_MODEL,
  );
  writeModelInput(
    dom.modelingModelInput,
    BASE_MODEL_PRESET.modelingModel,
    DEFAULT_MODELING_MODEL,
  );
  writeModelInput(
    dom.reviewModelInput,
    BASE_MODEL_PRESET.reviewModel,
    DEFAULT_REVIEW_MODEL,
  );
  writeModelInput(
    dom.nonTerminalEvaluationModelInput,
    BASE_MODEL_PRESET.nonTerminalEvaluationModel,
    DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
  );

  if (dom.timeoutInput instanceof HTMLInputElement) {
    dom.timeoutInput.value = profile.timeoutSeconds;
  }

  const scheduler = readJsonObjectOrFallback(dom.schedulerConfigInput.value, DEFAULT_SCHEDULER_CONFIG);
  dom.schedulerConfigInput.value = JSON.stringify(
    {
      ...scheduler,
      ...profile.scheduler,
      depth_preset: profile.key,
    },
    null,
    2,
  );

  const problemContext = readJsonObjectOrFallback(dom.problemContextInput.value, FALLBACK_DEFAULT_PROBLEM_CONTEXT);
  dom.problemContextInput.value = JSON.stringify(
    {
      ...problemContext,
      reasoning_depth_preset: profile.key,
    },
    null,
    2,
  );

  persistDraft();
  renderDepthPresetControls();
  setStatus(`${profile.label} preset applied.`, "ok");
}

function applyRecommendedModelPreset() {
  applyDepthPreset(RECOMMENDED_MODEL_PRESET.key);
}

function restoreDraft() {
  let stored = {};
  try {
    stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (_error) {
    stored = {};
  }

  const isLegacyDraft = stored.storageVersion !== STORAGE_VERSION;

  const fieldMap = {
    problemPromptInput: "problemPrompt",
    sessionIdInput: "sessionId",
    problemContextInput: "problemContext",
    schedulerConfigInput: "schedulerConfig",
    baseUrlInput: "baseUrl",
    deleteReasonInput: "deleteReason",
    steerPromptInput: "steerPrompt",
    searchInput: "searchQuery",
  };

  Object.entries(fieldMap).forEach(([domKey, storedKey]) => {
    const value = stored[storedKey];
    if (value !== undefined && dom[domKey]) {
      dom[domKey].value = String(value);
    }
  });

  if (isLegacyDraft) {
    dom.schedulerConfigInput.value = migrateLegacySchedulerConfig(dom.schedulerConfigInput.value);
  }

  const restoredTimeout = sanitizeTimeoutSeconds(
    isLegacyDraft ? migrateLegacyTimeoutSeconds(stored.timeout) : stored.timeout
  );
  dom.timeoutInput.value = String(restoredTimeout);

  writeModelInput(
    dom.planningModelInput,
    isLegacyDraft ? migrateLegacyPlanningModel(stored.planningModel) : stored.planningModel,
    DEFAULT_PLANNING_MODEL,
  );
  writeModelInput(
    dom.modelingModelInput,
    isLegacyDraft ? migrateLegacyModelingModel(stored.modelingModel) : stored.modelingModel,
    DEFAULT_MODELING_MODEL,
  );
  writeModelInput(
    dom.reviewModelInput,
    isLegacyDraft ? migrateLegacyReviewModel(stored.reviewModel) : stored.reviewModel,
    DEFAULT_REVIEW_MODEL,
  );
  writeModelInput(
    dom.nonTerminalEvaluationModelInput,
    LEGACY_NON_TERMINAL_EVALUATION_DEFAULTS.includes(
      sanitizeModelName(stored.nonTerminalEvaluationModel, DEFAULT_NON_TERMINAL_EVALUATION_MODEL),
    )
      ? DEFAULT_NON_TERMINAL_EVALUATION_MODEL
      : stored.nonTerminalEvaluationModel,
    DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
  );

  const restoredPollInterval = sanitizePollInterval(
    isLegacyDraft ? migrateLegacyPollInterval(stored.pollIntervalMs) : stored.pollIntervalMs
  );
  dom.pollIntervalInput.value = String(restoredPollInterval);

  if (typeof stored.pollingEnabled === "boolean") {
    dom.pollingToggle.checked = stored.pollingEnabled;
  }

  if (!dom.problemPromptInput.value.trim()) {
    const migratedProblemPrompt = extractProblemStatementDraft(dom.problemContextInput.value);
    if (migratedProblemPrompt) {
      dom.problemPromptInput.value = migratedProblemPrompt;
    }
  }

  uiState.sessionId = dom.sessionIdInput.value.trim();
  uiState.searchQuery = dom.searchInput.value.trim();
  uiState.pollingEnabled = dom.pollingToggle.checked;
  uiState.pollIntervalMs = restoredPollInterval;
  restartPolling();
  persistDraft();
}

function persistDraft() {
  const payload = {
    storageVersion: STORAGE_VERSION,
    problemPrompt: dom.problemPromptInput.value,
    sessionId: dom.sessionIdInput.value.trim(),
    problemContext: dom.problemContextInput.value,
    schedulerConfig: dom.schedulerConfigInput.value,
    baseUrl: dom.baseUrlInput.value.trim(),
    planningModel: readModelInput(dom.planningModelInput, DEFAULT_PLANNING_MODEL),
    modelingModel: readModelInput(dom.modelingModelInput, DEFAULT_MODELING_MODEL),
    reviewModel: readModelInput(dom.reviewModelInput, DEFAULT_REVIEW_MODEL),
    nonTerminalEvaluationModel: readModelInput(
      dom.nonTerminalEvaluationModelInput,
      DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    ),
    timeout: dom.timeoutInput.value.trim(),
    deleteReason: dom.deleteReasonInput.value,
    steerPrompt: dom.steerPromptInput.value,
    searchQuery: dom.searchInput.value,
    pollingEnabled: dom.pollingToggle.checked,
    pollIntervalMs: dom.pollIntervalInput.value.trim(),
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function migrateLegacySchedulerConfig(rawValue) {
  const fallback = JSON.stringify(DEFAULT_SCHEDULER_CONFIG, null, 2);
  if (!rawValue || !String(rawValue).trim()) {
    return fallback;
  }

  try {
    const parsed = JSON.parse(String(rawValue));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return fallback;
    }
    delete parsed.expansion_budget;
    if (parsed.max_tree_depth === undefined) {
      parsed.max_tree_depth = DEFAULT_SCHEDULER_CONFIG.max_tree_depth;
    }
    if (!parsed.depth_preset) {
      parsed.depth_preset = DEFAULT_SCHEDULER_CONFIG.depth_preset;
    } else {
      parsed.depth_preset = normalizeDepthPreset(parsed.depth_preset);
    }
    const presetProfile = DEPTH_PRESET_PROFILES[parsed.depth_preset] || DEPTH_PRESET_PROFILES[DEFAULT_DEPTH_PRESET];
    if (parsed.max_frontier_size === undefined || parsed.max_frontier_size === 8) {
      parsed.max_frontier_size = DEFAULT_SCHEDULER_CONFIG.max_frontier_size;
    }
    const legacyPresetChildSurface = {
      medium: 6,
      high: 8,
    };
    if (
      parsed.max_children_per_expansion === undefined
      || parsed.max_children_per_expansion === legacyPresetChildSurface[parsed.depth_preset]
    ) {
      parsed.max_children_per_expansion = presetProfile.scheduler.max_children_per_expansion;
    }
    if (parsed.max_live_children_per_batch === undefined) {
      parsed.max_live_children_per_batch = presetProfile.scheduler.max_live_children_per_batch;
    }
    if (parsed.use_local_root_proposal === undefined) {
      parsed.use_local_root_proposal = presetProfile.scheduler.use_local_root_proposal;
    }
    if (parsed.use_local_root_evaluation === undefined) {
      parsed.use_local_root_evaluation = presetProfile.scheduler.use_local_root_evaluation;
    }
    if (parsed.use_local_child_proposal === undefined) {
      parsed.use_local_child_proposal = presetProfile.scheduler.use_local_child_proposal;
    }
    if (parsed.use_local_child_evaluation === undefined) {
      parsed.use_local_child_evaluation = presetProfile.scheduler.use_local_child_evaluation;
    }
    if (parsed.max_frontier_per_diversity_key === undefined || parsed.max_frontier_per_diversity_key === 2) {
      parsed.max_frontier_per_diversity_key = DEFAULT_SCHEDULER_CONFIG.max_frontier_per_diversity_key;
    }
    return JSON.stringify(parsed, null, 2);
  } catch (_error) {
    return fallback;
  }
}

function wireEvents() {
  dom.attachSessionButton.addEventListener("click", () => {
    runUiAction(() => attachSession());
  });
  dom.createSessionButton.addEventListener("click", () => {
    runUiAction(() => createSession());
  });
  dom.refreshSessionButton.addEventListener("click", () => {
    runUiAction(() => refreshSession());
  });
  dom.exportAnswerButton.addEventListener("click", () => {
    exportCurrentAnswer();
  });
  dom.dropSessionButton.addEventListener("click", () => {
    runUiAction(() => dropSession());
  });
  dom.prevMatchButton.addEventListener("click", () => {
    jumpToSearchMatch(-1);
  });
  dom.nextMatchButton.addEventListener("click", () => {
    jumpToSearchMatch(1);
  });
  dom.expandAllButton.addEventListener("click", () => {
    uiState.collapsedNodeIds.clear();
    uiState.revealSelection = true;
    render();
    setStatus("Expanded all visible branches.", "ok");
  });
  dom.collapseAllButton.addEventListener("click", () => {
    collapseAllDescendants();
  });
  dom.deleteNodeButton.addEventListener("click", () => {
    runUiAction(() => deleteSelectedNode());
  });
  dom.deleteSteerRunButton.addEventListener("click", () => {
    runUiAction(() => deleteSelectedNode({ steer: true, runAfterDelete: true }));
  });
  [
    [dom.applyLowDepthPresetButton, "low"],
    [dom.applyMediumDepthPresetButton, "medium"],
    [dom.applyHighDepthPresetButton, "high"],
  ].forEach(([button, presetName]) => {
    if (button instanceof HTMLButtonElement) {
      button.addEventListener("click", () => {
        applyDepthPreset(presetName);
      });
    }
  });

  dom.sessionIdInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      runUiAction(() => attachSession());
    }
  });

  dom.searchInput.addEventListener("input", () => {
    uiState.searchQuery = dom.searchInput.value.trim();
    persistDraft();
    render();
  });
  dom.searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      jumpToSearchMatch(event.shiftKey ? -1 : 1);
    }
    if (event.key === "Escape") {
      event.preventDefault();
      dom.treeViewport.focus();
    }
  });

  dom.pollingToggle.addEventListener("change", () => {
    uiState.pollingEnabled = dom.pollingToggle.checked;
    restartPolling();
    persistDraft();
    setStatus(
      uiState.pollingEnabled ? "Auto refresh enabled." : "Auto refresh disabled.",
      "ok"
    );
    render();
  });

  dom.pollIntervalInput.addEventListener("change", () => {
    uiState.pollIntervalMs = sanitizePollInterval(dom.pollIntervalInput.value);
    dom.pollIntervalInput.value = String(uiState.pollIntervalMs);
    restartPolling();
    persistDraft();
    render();
  });

  [
    dom.sessionIdInput,
    dom.problemPromptInput,
    dom.problemContextInput,
    dom.schedulerConfigInput,
    dom.baseUrlInput,
    dom.planningModelInput,
    dom.modelingModelInput,
    dom.reviewModelInput,
    dom.nonTerminalEvaluationModelInput,
    dom.timeoutInput,
    dom.deleteReasonInput,
    dom.steerPromptInput,
  ].filter(Boolean).forEach((element) => {
    const eventName = element instanceof HTMLInputElement && element.type === "checkbox" ? "change" : "input";
    element.addEventListener(eventName, persistDraft);
  });

  dom.treeLines.addEventListener("click", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (!(target instanceof HTMLElement)) {
      return;
    }
    selectNode(target.dataset.nodeId || "");
    dom.treeViewport.focus();
  });

  dom.treeLines.addEventListener("dblclick", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (!(target instanceof HTMLElement)) {
      return;
    }
    toggleSelectedNodeExpansion(target.dataset.nodeId || "");
    dom.treeViewport.focus();
  });

  dom.frontierList.addEventListener("click", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (!(target instanceof HTMLElement)) {
      return;
    }
    selectNode(target.dataset.nodeId || "");
    dom.treeViewport.focus();
  });

  dom.bestResultCard.addEventListener("click", () => {
    const nodeId = dom.bestResultCard.dataset.nodeId || "";
    if (!nodeId) {
      return;
    }
    selectNode(nodeId);
    dom.treeViewport.focus();
  });

  dom.candidateResults.addEventListener("click", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (!(target instanceof HTMLElement)) {
      return;
    }
    selectNode(target.dataset.nodeId || "");
    dom.treeViewport.focus();
  });

  document.addEventListener("keydown", handleGlobalKeydown);
}

function handleGlobalKeydown(event) {
  if ((event.shiftKey || event.altKey) && event.key.startsWith("Arrow")) {
    event.preventDefault();
    panViewport(event.key);
    return;
  }

  if (isTextEditingTarget(event.target)) {
    if (event.key === "Escape") {
      dom.treeViewport.focus();
    }
    return;
  }

  switch (event.key) {
    case "/":
      event.preventDefault();
      dom.searchInput.focus();
      dom.searchInput.select();
      break;
    case "ArrowUp":
      event.preventDefault();
      moveSelection(-1);
      break;
    case "ArrowDown":
      event.preventDefault();
      moveSelection(1);
      break;
    case "ArrowLeft":
      event.preventDefault();
      navigateLeft();
      break;
    case "ArrowRight":
      event.preventDefault();
      navigateRight();
      break;
    case "Home":
      event.preventDefault();
      if (uiState.treeModel.root) {
        selectNode(uiState.treeModel.root.id);
      }
      break;
    case "End":
      event.preventDefault();
      if (uiState.treeModel.visibleNodes.length) {
        selectNode(uiState.treeModel.visibleNodes[uiState.treeModel.visibleNodes.length - 1].node.id);
      }
      break;
    case "Enter":
      if (uiState.searchQuery) {
        event.preventDefault();
        jumpToSearchMatch(event.shiftKey ? -1 : 1);
      }
      break;
    case "r":
    case "R":
      event.preventDefault();
      runUiAction(() => refreshSession());
      break;
    case "p":
    case "P":
      event.preventDefault();
      dom.pollingToggle.checked = !dom.pollingToggle.checked;
      dom.pollingToggle.dispatchEvent(new Event("change"));
      break;
    case "e":
    case "E":
      event.preventDefault();
      runUiAction(() => runSession());
      break;
    case "Delete":
    case "Backspace":
      event.preventDefault();
      runUiAction(() => deleteSelectedNode());
      break;
    default:
      break;
  }
}

function isTextEditingTarget(target) {
  return target instanceof HTMLElement && (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT" ||
    target.isContentEditable
  );
}

function sanitizePollInterval(rawValue) {
  const parsed = Number.parseInt(String(rawValue || DEFAULT_POLL_INTERVAL_MS), 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_POLL_INTERVAL_MS;
  }
  return Math.max(DEFAULT_POLL_INTERVAL_MS, parsed);
}

function sanitizeTimeoutSeconds(rawValue) {
  const parsed = Number.parseFloat(String(rawValue || DEFAULT_TIMEOUT_SECONDS));
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_TIMEOUT_SECONDS;
  }
  return parsed;
}

function migrateLegacyTimeoutSeconds(rawValue) {
  const parsed = Number.parseFloat(String(rawValue || ""));
  if (!Number.isFinite(parsed) || parsed === 10 || parsed === 30 || parsed === 60 || parsed === 120) {
    return DEFAULT_TIMEOUT_SECONDS;
  }
  return parsed;
}

function migrateLegacyPollInterval(rawValue) {
  const parsed = Number.parseInt(String(rawValue || ""), 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_POLL_INTERVAL_MS;
  }
  if (parsed === 60_000) {
    return DEFAULT_POLL_INTERVAL_MS;
  }
  return Math.max(DEFAULT_POLL_INTERVAL_MS, parsed);
}

function formatPollInterval(rawValue) {
  const intervalMs = sanitizePollInterval(rawValue);
  if (intervalMs % 60_000 === 0) {
    const minutes = intervalMs / 60_000;
    return `${minutes}min`;
  }
  if (intervalMs % 1000 === 0) {
    return `${intervalMs / 1000}s`;
  }
  return `${intervalMs}ms`;
}

function restartPolling() {
  if (uiState.pollTimer) {
    window.clearInterval(uiState.pollTimer);
    uiState.pollTimer = null;
  }

  if (!uiState.pollingEnabled) {
    return;
  }

  uiState.pollTimer = window.setInterval(() => {
    if (!uiState.sessionId || uiState.requestInFlight) {
      return;
    }
    refreshSession({ silent: true });
  }, uiState.pollIntervalMs);
}

function enableAutoRefreshForSession() {
  if (!uiState.sessionId) {
    return;
  }

  uiState.pollingEnabled = true;
  restartPolling();
  persistDraft();
}

function clearBusyRefresh() {
  if (uiState.busyRefreshTimer) {
    window.clearTimeout(uiState.busyRefreshTimer);
    uiState.busyRefreshTimer = null;
  }
}

function scheduleBusyRefresh() {
  clearBusyRefresh();
  if (!uiState.sessionId) {
    return;
  }

  uiState.busyRefreshTimer = window.setTimeout(() => {
    uiState.busyRefreshTimer = null;
    if (!uiState.sessionId || uiState.requestInFlight) {
      return;
    }
    refreshSession({ silent: true });
  }, BUSY_REFRESH_INTERVAL_MS);
}

function setStatus(message, tone = "idle", options = {}) {
  uiState.statusMessage = String(message || "").trim() || "SYSTEM READY // no session attached";
  uiState.statusTone = String(tone || "idle");
  if (options.record !== false) {
    pushStatusLog(uiState.statusMessage, "", tone);
  }
  renderStatus();
  renderStatusLog();
}

function setActionFeedback(title, detail, tone = "idle", options = {}) {
  uiState.lastActionTitle = String(title || "").trim() || "No request yet.";
  uiState.lastActionDetail = String(detail || "").trim() || "No further details.";
  uiState.lastActionTone = tone;
  if (options.record !== false) {
    pushStatusLog(uiState.lastActionTitle, uiState.lastActionDetail, tone);
  }
  renderActionFeedback();
  renderStatusLog();
}

function pushStatusLog(title, detail, tone = "idle") {
  const normalizedTitle = String(title || "").trim() || "No title";
  const normalizedDetail = String(detail || "").trim();
  const normalizedTone = String(tone || "idle");
  const timestamp = new Date();
  const latestEntry = uiState.statusLogEntries[0] || null;

  if (
    latestEntry &&
    latestEntry.title === normalizedTitle &&
    latestEntry.detail === normalizedDetail &&
    latestEntry.tone === normalizedTone
  ) {
    latestEntry.count += 1;
    latestEntry.at = timestamp.toISOString();
    return;
  }

  uiState.statusLogSequence += 1;
  uiState.statusLogEntries.unshift({
    id: `status-${uiState.statusLogSequence}`,
    title: normalizedTitle,
    detail: normalizedDetail,
    tone: normalizedTone,
    at: timestamp.toISOString(),
    count: 1,
  });
  uiState.statusLogEntries = uiState.statusLogEntries.slice(0, 40);
}

function captureScrollPosition(element) {
  if (!(element instanceof HTMLElement)) {
    return null;
  }
  return {
    top: element.scrollTop,
    left: element.scrollLeft,
  };
}

function restoreScrollPosition(element, state) {
  if (!(element instanceof HTMLElement) || !state) {
    return;
  }
  element.scrollTop = state.top;
  element.scrollLeft = state.left;
}

function captureScrollState() {
  return {
    treeViewport: captureScrollPosition(dom.treeViewport),
    detailBody: captureScrollPosition(dom.detailBody),
    frontierList: captureScrollPosition(dom.frontierList),
    activityLog: captureScrollPosition(dom.activityLog),
    statusLog: captureScrollPosition(dom.statusLog),
  };
}

function restoreScrollState(scrollState, options = {}) {
  if (options.restoreTree !== false) {
    restoreScrollPosition(dom.treeViewport, scrollState.treeViewport);
  }
  restoreScrollPosition(dom.detailBody, scrollState.detailBody);
  restoreScrollPosition(dom.frontierList, scrollState.frontierList);
  restoreScrollPosition(dom.activityLog, scrollState.activityLog);
  restoreScrollPosition(dom.statusLog, scrollState.statusLog);
}

function render() {
  const scrollState = captureScrollState();
  const shouldRevealSelection = uiState.revealSelection;
  recomputeDerivedState();
  renderStatus();
  renderActionFeedback();
  renderStatusLog();
  renderButtons();
  renderDepthPresetControls();
  renderResultsBoard();
  renderTree();
  renderDetailPane();
  renderFrontier();
  renderActivity();
  renderMeta();
  restoreScrollState(scrollState, { restoreTree: !shouldRevealSelection });
}

function recomputeDerivedState() {
  const root = getRenderableRoot(uiState.snapshot);
  if (!root) {
    uiState.treeModel = emptyTreeModel();
    uiState.searchMatches = [];
    uiState.searchCursor = 0;
    return;
  }

  let model = buildTreeModel(root, uiState.collapsedNodeIds);
  Array.from(uiState.collapsedNodeIds).forEach((nodeId) => {
    if (!model.nodeById.has(nodeId)) {
      uiState.collapsedNodeIds.delete(nodeId);
    }
  });

  if (!uiState.selectedNodeId || !model.nodeById.has(uiState.selectedNodeId)) {
    uiState.selectedNodeId = root.id;
    uiState.revealSelection = true;
  }

  const ancestorIds = model.ancestorsById.get(uiState.selectedNodeId) || [];
  ancestorIds.forEach((ancestorId) => uiState.collapsedNodeIds.delete(ancestorId));

  model = buildTreeModel(root, uiState.collapsedNodeIds);
  uiState.treeModel = model;
  uiState.searchMatches = computeSearchMatches(model.allNodes, uiState.searchQuery);

  const currentMatchIndex = uiState.searchMatches.indexOf(uiState.selectedNodeId);
  if (currentMatchIndex >= 0) {
    uiState.searchCursor = currentMatchIndex;
  } else if (uiState.searchMatches.length === 0) {
    uiState.searchCursor = 0;
  } else {
    uiState.searchCursor = Math.min(uiState.searchCursor, uiState.searchMatches.length - 1);
  }
}

function getRunState(snapshot) {
  const runState = snapshot && isPlainObject(snapshot.run_state) ? snapshot.run_state : {};
  return isPlainObject(runState) ? runState : {};
}

function getInFlightExpansion(snapshot) {
  const inFlight = getRunState(snapshot).in_flight_expansion;
  return isPlainObject(inFlight) ? inFlight : null;
}

function getInFlightExpansionMetrics(snapshot) {
  const inFlight = getInFlightExpansion(snapshot);
  if (!inFlight) {
    return null;
  }
  const parentId = String(inFlight.parent_id || "").trim();
  const parentDepth = Number.isFinite(Number(inFlight.parent_depth))
    ? Number(inFlight.parent_depth)
    : null;
  const expectedChildCount = Number.isFinite(Number(inFlight.expected_child_count))
    ? Number(inFlight.expected_child_count)
    : 0;
  const builtChildCount = Number.isFinite(Number(inFlight.built_child_count))
    ? Number(inFlight.built_child_count)
    : 0;
  return {
    parentId,
    parentDepth,
    expectedChildCount,
    builtChildCount,
  };
}

function formatInFlightExpansion(snapshot, { compact = false } = {}) {
  const metrics = getInFlightExpansionMetrics(snapshot);
  if (!metrics) {
    return "";
  }
  const parentHint = metrics.parentId ? metrics.parentId.slice(0, 8) : "pending";
  if (compact) {
    return `in-flight ${metrics.builtChildCount}/${metrics.expectedChildCount}`;
  }
  const depthLabel = metrics.parentDepth === null ? "" : ` at depth ${metrics.parentDepth}`;
  return `${metrics.builtChildCount}/${metrics.expectedChildCount} child nodes built for ${parentHint}${depthLabel}`;
}

function isSessionBusy(snapshot) {
  const status = String(getRunState(snapshot).status || "idle").trim().toLowerCase();
  return status === "busy";
}

function getRenderableRoot(snapshot) {
  const root = snapshot && snapshot.root ? snapshot.root : null;
  if (root) {
    return root;
  }

  const runState = getRunState(snapshot);
  const metaTask = snapshot && typeof snapshot.meta_task === "object" && snapshot.meta_task
    ? snapshot.meta_task
    : {};
  const hasPendingSession = Boolean(
    snapshot
    && (
      Object.keys(runState).length > 0
      || Object.keys(metaTask).length > 0
      || uiState.sessionId
    )
  );
  if (!hasPendingSession) {
    return null;
  }

  const phase = String(runState.phase || "created").trim() || "created";
  const status = String(runState.status || "idle").trim() || "idle";
  const lastError = String(runState.last_error || "").trim();
  const objective = String(metaTask.objective || "").trim();
  const firstStep = String(metaTask.first_step || "").trim();
  const displayStatus = status.toLowerCase() === "error" ? "ERROR" : "PENDING";
  const waitingSummary = status.toLowerCase() === "error"
    ? `Root construction failed before the first real branch was created.${lastError ? ` Last error: ${lastError}.` : ""}`
    : firstStep
      ? `Waiting for the first root branch. Planned first step: ${firstStep}.`
      : `Waiting for the first root branch. Backend phase: ${phase}.`;

  return {
    id: uiState.sessionId ? `session-${uiState.sessionId.slice(0, 8)}` : "session-pending",
    parent_id: null,
    thought_step: objective ? `${objective} ${waitingSummary}`.trim() : waitingSummary,
    equations: [],
    known_vars: {
      synthetic_placeholder_root: true,
      run_status: status,
      run_phase: phase,
      last_error: lastError,
      meta_task_objective: objective,
      meta_task_first_step: firstStep,
    },
    used_models: [],
    quantities: {},
    boundary_conditions: {},
    status: displayStatus,
    fsm_state: "PENDING ROOT",
    score: 0,
    reflection_history: [],
    children: [],
  };
}

function isSyntheticPlaceholderRoot(node) {
  return Boolean(node && node.known_vars && node.known_vars.synthetic_placeholder_root);
}

function buildTreeModel(root, collapsedNodeIds) {
  const model = emptyTreeModel();
  model.root = root;

  function walk(node, depth, ancestorIds, branchGuides, isLast) {
    const children = getChildren(node);
    const isCollapsed = children.length > 0 && collapsedNodeIds.has(node.id);
    const summary = buildNodePresentation(node, children.length, isCollapsed, depth);
    const searchText = buildSearchText(node, depth);

    model.nodeById.set(node.id, node);
    model.ancestorsById.set(node.id, ancestorIds);
    model.allNodes.push({
      node,
      depth,
      ancestorIds,
      searchText,
      childCount: children.length,
    });
    model.visibleNodes.push({
      node,
      depth,
      ancestorIds,
      childCount: children.length,
      isCollapsed,
      summary,
    });

    model.maxDepth = Math.max(model.maxDepth, depth);
    if (children.length === 0) {
      model.leafCount += 1;
    }
    if (String(node.status || "").toUpperCase().startsWith("PRUNED")) {
      model.prunedCount += 1;
    } else {
      model.activeCount += 1;
    }

    if (isCollapsed) {
      return;
    }

    children.forEach((child, index) => {
      walk(
        child,
        depth + 1,
        ancestorIds.concat(node.id),
        branchGuides.concat(index < children.length - 1),
        index === children.length - 1,
      );
    });
  }

  walk(root, 0, [], [], true);
  return model;
}

function getChildren(node) {
  return Array.isArray(node && node.children) ? node.children : [];
}

function buildNodePresentation(node, childCount, isCollapsed, depth) {
  const foldMark = depth === 0 ? "root" : childCount > 0 ? (isCollapsed ? "+" : "-") : "•";
  const visibleResult = displayResultState(node);
  return {
    foldMark,
    title: summarizeThought(node.thought_step),
    meta: buildNodeMeta(node, childCount, depth),
    routeFamily: getNodeRouteFamily(node),
    stepFocus: getNodeStepFocus(node),
    ignoredNoiseCount: getNodeIgnoredNoiseCount(node),
    status: shortStatus(visibleResult),
    statusTone: statusTone(visibleResult),
  };
}

function buildNodeMeta(node, childCount, depth) {
  const parts = [
    node.id,
    `d${depth}`,
    childCount === 0 ? "leaf" : `${childCount} child${childCount === 1 ? "" : "ren"}`,
  ];

  const resultState = displayResultState(node);
  if (resultState) {
    parts.push(String(resultState).toLowerCase());
  }
  if (Number.isFinite(node.score)) {
    parts.push(`score ${formatScore(node.score)}`);
  }
  return parts.join(" · ");
}

function summarizeThought(thoughtStep) {
  const normalized = trimText(String(thoughtStep || "").trim(), 72);
  return normalized || "No thought step recorded.";
}

function getNodeRouteFamily(node) {
  if (!node || !node.known_vars || typeof node.known_vars !== "object") {
    return "";
  }
  const direct = String(node.known_vars.route_family || "").trim();
  if (direct) {
    return direct;
  }
  const task = node.known_vars.orchestrator_task;
  if (task && typeof task === "object") {
    return String(task.selected_route_family || "").trim();
  }
  return "";
}

function getNodeStepFocus(node) {
  if (!node || !node.known_vars || typeof node.known_vars !== "object") {
    return "";
  }
  const task = node.known_vars.orchestrator_task;
  if (!task || typeof task !== "object") {
    return "";
  }
  return trimText(String(task.step_focus || task.selected_task || "").trim(), 26);
}

function getNodeIgnoredNoiseCount(node) {
  if (!node || !node.known_vars || typeof node.known_vars !== "object") {
    return 0;
  }
  let count = 0;
  if (Array.isArray(node.known_vars.ignored_review_rule_violations)) {
    count += node.known_vars.ignored_review_rule_violations.length;
  }
  const hardRuleCheck = node.known_vars.hard_rule_check;
  if (hardRuleCheck && typeof hardRuleCheck === "object" && Array.isArray(hardRuleCheck.ignored_violations)) {
    count += hardRuleCheck.ignored_violations.length;
  }
  return count;
}

function buildSearchText(node, depth) {
  return [
    node.id,
    node.parent_id,
    node.status,
    node.fsm_state,
    node.thought_step,
    depth,
    joinStrings(node.equations),
    safeJson(node.known_vars),
    joinStrings(node.used_models),
    safeJson(node.quantities),
    safeJson(node.boundary_conditions),
    joinStrings(node.reflection_history),
  ]
    .join(" ")
    .toLowerCase();
}

function computeSearchMatches(allNodes, query) {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  if (!normalizedQuery) {
    return [];
  }
  return allNodes
    .filter((entry) => entry.searchText.includes(normalizedQuery))
    .map((entry) => entry.node.id);
}

function renderStatus() {
  const stamp = uiState.lastUpdatedAt ? ` | updated ${formatClock(uiState.lastUpdatedAt)}` : "";
  const session = uiState.sessionId ? ` | session ${uiState.sessionId}` : "";
  const polling = ` | poll ${uiState.pollingEnabled ? formatPollInterval(uiState.pollIntervalMs) : "off"}`;
  dom.statusLine.textContent = `${uiState.statusMessage}${session}${polling}${stamp}`;
  dom.statusLine.className = `status-line status-${uiState.statusTone}`;
}

function renderActionFeedback() {
  dom.actionFeedbackTitle.textContent = uiState.lastActionTitle;
  dom.actionFeedbackDetail.textContent = uiState.lastActionDetail;
  dom.actionFeedback.className = `action-feedback status-${uiState.lastActionTone}`;
}

function renderStatusLog() {
  if (!dom.statusLog) {
    return;
  }

  if (!uiState.statusLogEntries.length) {
    const empty = document.createElement("div");
    empty.className = "status-log-empty";
    empty.textContent = "No status history yet.";
    dom.statusLog.replaceChildren(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  uiState.statusLogEntries.forEach((entry) => {
    const item = document.createElement("div");
    item.className = `status-log-entry status-${entry.tone}`;

    const header = document.createElement("div");
    header.className = "status-log-header";

    const time = document.createElement("span");
    time.className = "status-log-time";
    time.textContent = formatClock(new Date(entry.at));

    const badge = document.createElement("span");
    badge.className = `status-log-badge status-${entry.tone}`;
    badge.textContent = shortUiTone(entry.tone);

    const title = document.createElement("div");
    title.className = "status-log-title";
    title.textContent = entry.title;

    header.append(time, badge, title);

    if (entry.count > 1) {
      const count = document.createElement("span");
      count.className = "status-log-count";
      count.textContent = `x${entry.count}`;
      header.append(count);
    }

    item.append(header);

    if (entry.detail) {
      const detail = document.createElement("div");
      detail.className = "status-log-detail";
      detail.textContent = entry.detail;
      item.append(detail);
    }

    fragment.append(item);
  });

  dom.statusLog.replaceChildren(fragment);
}

function renderButtons() {
  const hasSession = Boolean(uiState.sessionId);
  const hasSelection = Boolean(getSelectedNode());
  const rootNode = uiState.treeModel.root;
  const selectionIsRoot = Boolean(rootNode && uiState.selectedNodeId === rootNode.id);
  const sessionBusy = isSessionBusy(uiState.snapshot);

  [
    dom.attachSessionButton,
    dom.createSessionButton,
    dom.refreshSessionButton,
    dom.exportAnswerButton,
    dom.dropSessionButton,
    dom.prevMatchButton,
    dom.nextMatchButton,
    dom.expandAllButton,
    dom.collapseAllButton,
    dom.deleteNodeButton,
    dom.deleteSteerRunButton,
    dom.applyLowDepthPresetButton,
    dom.applyMediumDepthPresetButton,
    dom.applyHighDepthPresetButton,
  ].forEach((button) => {
    button.disabled = uiState.requestInFlight;
  });

  dom.refreshSessionButton.disabled = uiState.requestInFlight || !hasSession;
  dom.dropSessionButton.disabled = uiState.requestInFlight || !hasSession;
  const resultsBoard = buildResultsBoard(uiState.treeModel);
  dom.exportAnswerButton.disabled =
    uiState.requestInFlight || !shouldShowResultsBoard(uiState.snapshot, resultsBoard);
  dom.expandAllButton.disabled = !uiState.snapshot;
  dom.collapseAllButton.disabled = !uiState.snapshot;
  dom.prevMatchButton.disabled = uiState.searchMatches.length === 0;
  dom.nextMatchButton.disabled = uiState.searchMatches.length === 0;
  dom.deleteNodeButton.disabled =
    uiState.requestInFlight || !hasSession || !hasSelection || selectionIsRoot;
  dom.deleteSteerRunButton.disabled =
    uiState.requestInFlight || !hasSession || !hasSelection || selectionIsRoot || sessionBusy;
}

function renderMeta() {
  const model = uiState.treeModel;
  const frontier = Array.isArray(uiState.snapshot && uiState.snapshot.frontier)
    ? uiState.snapshot.frontier
    : [];
  const runState = getRunState(uiState.snapshot);
  const inFlightLabel = formatInFlightExpansion(uiState.snapshot, { compact: true });
  const inFlightSuffix = inFlightLabel ? ` | ${inFlightLabel}` : "";

  if (!model.root) {
    dom.treeStats.textContent = "no tree loaded";
    dom.selectionStats.textContent = "selection: none";
    dom.searchMeta.textContent = uiState.searchQuery ? "0 matches" : "0 matches";
    return;
  }

  if (model.root.known_vars && model.root.known_vars.synthetic_placeholder_root) {
    dom.treeStats.textContent =
      `root pending | status ${String(runState.status || "idle")} | phase ${String(runState.phase || "created")} | frontier ${frontier.length}${inFlightSuffix}`;
    dom.selectionStats.textContent = "selection: pending root";
    dom.searchMeta.textContent = uiState.searchQuery ? "0 matches" : "0 matches";
    return;
  }

  dom.treeStats.textContent =
    `nodes ${model.allNodes.length} | active ${model.activeCount} | pruned ${model.prunedCount} | leaves ${model.leafCount} | depth ${model.maxDepth} | frontier ${frontier.length}${inFlightSuffix}`;

  const selectedNode = getSelectedNode();
  if (selectedNode) {
    const selectedEntry = model.allNodes.find((entry) => entry.node.id === selectedNode.id);
    const selectedMatchIndex = uiState.searchMatches.indexOf(selectedNode.id);
    const childCount = getChildren(selectedNode).length;
    const routeFamily = getNodeRouteFamily(selectedNode);
    const stepFocus = getNodeStepFocus(selectedNode);
    const ignoredNoiseCount = getNodeIgnoredNoiseCount(selectedNode);
    const matchLabel = uiState.searchMatches.length
      ? ` | match ${selectedMatchIndex >= 0 ? selectedMatchIndex + 1 : 0}/${uiState.searchMatches.length}`
      : "";
    const routeLabel = routeFamily ? ` | route ${routeFamily}` : "";
    const stepLabel = stepFocus ? ` | step ${stepFocus}` : "";
    const noiseLabel = ignoredNoiseCount ? ` | noise ${ignoredNoiseCount}` : "";
    dom.selectionStats.textContent =
      `selection ${selectedNode.id} | depth ${selectedEntry ? selectedEntry.depth : 0} | children ${childCount}${routeLabel}${stepLabel}${noiseLabel}${matchLabel}`;
  } else {
    dom.selectionStats.textContent = "selection: none";
  }

  if (!uiState.searchQuery) {
    dom.searchMeta.textContent = "0 matches";
  } else if (uiState.searchMatches.length === 0) {
    dom.searchMeta.textContent = `0 matches for \"${uiState.searchQuery}\"`;
  } else {
    dom.searchMeta.textContent =
      `${uiState.searchMatches.length} matches | current ${uiState.searchCursor + 1}/${uiState.searchMatches.length}`;
  }
}

function shouldShowResultsBoard(snapshot, board) {
  if (!uiState.sessionId || !snapshot || !board || !board.best) {
    return false;
  }
  if (uiState.requestInFlight || uiState.busyRefreshTimer) {
    return false;
  }
  if (isSessionBusy(snapshot)) {
    return false;
  }
  return !uiState.pollingEnabled;
}

function renderResultsBoard() {
  const board = buildResultsBoard(uiState.treeModel);
  const showResultsBoard = shouldShowResultsBoard(uiState.snapshot, board);
  if (dom.resultsPanel) {
    dom.resultsPanel.hidden = !showResultsBoard;
  }
  if (!showResultsBoard) {
    return;
  }

  const runState = getRunState(uiState.snapshot);
  const phase = String(runState.phase || "created").trim();
  const runLabel = isSessionBusy(uiState.snapshot)
    ? `busy ${phase}`
    : phase || "idle";

  if (!board.best) {
    dom.resultSummary.textContent = `no scored candidates | ${runLabel}`;
    dom.bestResultCard.disabled = true;
    dom.bestResultCard.dataset.nodeId = "";
    dom.bestResultScore.textContent = "-";
    dom.bestResultMeta.textContent = "no result";
    dom.bestResultFormula.textContent = "No formula yet.";
    const empty = document.createElement("div");
    empty.className = "candidate-empty";
    empty.textContent = "No candidates.";
    dom.candidateResults.replaceChildren(empty);
    return;
  }

  const best = board.best;
  dom.resultSummary.textContent =
    `${board.label} ${board.scoredCount} | best score ${formatScore(best.node.score)} | ${runLabel}`;
  dom.bestResultCard.disabled = false;
  dom.bestResultCard.dataset.nodeId = best.node.id;
  dom.bestResultScore.textContent = formatScore(best.node.score);
  dom.bestResultMeta.textContent = buildResultMeta(best);
  dom.bestResultFormula.textContent = buildNodeAnswerText(best.node, { maxLength: 1_400 });

  const fragment = document.createDocumentFragment();
  board.candidates.forEach((entry, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-result";
    if (entry.node.id === best.node.id) {
      button.classList.add("candidate-result-best");
    }
    if (entry.node.id === uiState.selectedNodeId) {
      button.classList.add("selected");
    }
    button.dataset.nodeId = entry.node.id;

    const rank = document.createElement("span");
    rank.className = "candidate-rank";
    rank.textContent = String(index + 1).padStart(2, "0");

    const body = document.createElement("span");
    body.className = "candidate-body";

    const meta = document.createElement("span");
    meta.className = "candidate-meta";
    meta.textContent = `score ${formatScore(entry.node.score)} | depth ${entry.depth}`;

    const formula = document.createElement("span");
    formula.className = "candidate-formula";
    formula.textContent = buildNodeAnswerText(entry.node, { maxLength: 260 });

    body.append(meta, formula);
    button.append(rank, body);
    fragment.append(button);
  });

  dom.candidateResults.replaceChildren(fragment);
}

function buildResultsBoard(model) {
  const entries = Array.isArray(model && model.allNodes) ? model.allNodes : [];
  const scoredEntries = entries.filter((entry) => isScoredResultCandidate(entry));
  const nonRootEntries = scoredEntries.filter((entry) => entry.depth > 0);
  const leafEntries = nonRootEntries.filter((entry) => entry.childCount === 0);
  const sourceEntries = leafEntries.length ? leafEntries : nonRootEntries.length ? nonRootEntries : scoredEntries;
  const sorted = sourceEntries.slice().sort(compareResultEntries);
  return {
    best: sorted[0] || null,
    candidates: sorted.slice(0, RESULT_CANDIDATE_LIMIT),
    scoredCount: sorted.length,
    label: leafEntries.length ? "final leaves" : "current candidates",
  };
}

function isScoredResultCandidate(entry) {
  const node = entry && entry.node ? entry.node : null;
  if (!node || isSyntheticPlaceholderRoot(node) || !Number.isFinite(Number(node.score))) {
    return false;
  }
  const status = String(node.status || "").trim().toUpperCase();
  if (status.startsWith("PRUNED")) {
    return false;
  }
  return displayResultState(node) !== "DROP";
}

function compareResultEntries(left, right) {
  const leftScore = Number(left.node.score);
  const rightScore = Number(right.node.score);
  if (rightScore !== leftScore) {
    return rightScore - leftScore;
  }
  if (right.depth !== left.depth) {
    return right.depth - left.depth;
  }
  if (left.childCount !== right.childCount) {
    return left.childCount - right.childCount;
  }
  return String(left.node.id).localeCompare(String(right.node.id));
}

function buildResultMeta(entry) {
  const node = entry.node;
  const parts = [
    `depth ${entry.depth}`,
    shortStatus(displayResultState(node)),
  ];
  const routeFamily = getNodeRouteFamily(node);
  const stepFocus = getNodeStepFocus(node);
  if (routeFamily) {
    parts.push(routeFamily);
  }
  if (stepFocus) {
    parts.push(stepFocus);
  }
  return parts.join(" | ");
}

function summarizeResultThought(node) {
  return trimText(String(node && node.thought_step ? node.thought_step : "No thought step recorded."), 180);
}

function buildNodeAnswerText(node, options = {}) {
  const maxLength = Number.isFinite(options.maxLength) ? Number(options.maxLength) : 1_000;
  const bestCandidate = selectBestAnswerCandidate(collectNodeAnswerCandidates(node));
  if (bestCandidate && isConcreteAnswerCandidate(bestCandidate)) {
    return trimText(bestCandidate.text, maxLength);
  }
  return trimText(NO_CONCRETE_FINAL_ANSWER_TEXT, maxLength);
}

function collectNodeAnswerCandidates(node) {
  const candidates = [];
  const equationLines = collectConcreteEquationLines(node && node.equations);
  if (equationLines.length) {
    pushAnswerCandidate(candidates, equationLines.join("\n"), { source: "equations" });
  }

  collectAnswerCandidates(candidates, node && node.known_vars, { source: "known_vars" });
  collectAnswerCandidates(candidates, node && node.quantities, { source: "quantities" });
  pushAnswerCandidate(candidates, formatStructuredAnswer(node && node.quantities), {
    source: "quantities-summary",
  });
  collectAnswerCandidates(candidates, node && node.boundary_conditions, { source: "boundary_conditions" });
  pushAnswerCandidate(candidates, formatStructuredAnswer(node && node.boundary_conditions), {
    source: "boundary-summary",
  });
  pushAnswerCandidate(candidates, formatInlineValue(node && node.thought_step), { source: "thought_step" });

  return candidates;
}

function collectAnswerCandidates(candidates, value, context = {}, depth = 0) {
  if (value === null || value === undefined || depth > 2) {
    return;
  }

  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    if (context.key && isAnswerLikeKey(context.key)) {
      pushAnswerCandidate(candidates, formatInlineValue(value), context);
    }
    return;
  }

  if (Array.isArray(value)) {
    value.forEach((entry, index) => {
      collectAnswerCandidates(candidates, entry, { ...context, key: context.key || String(index) }, depth + 1);
    });
    return;
  }

  if (!isPlainObject(value)) {
    return;
  }

  for (const key of RESULT_ANSWER_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(value, key)) {
      continue;
    }
    const directValue = value[key];
    const directText = typeof directValue === "object" && directValue !== null
      ? formatStructuredAnswer(directValue)
      : formatInlineValue(directValue);
    pushAnswerCandidate(candidates, directText, { ...context, key, direct: true });
  }

  for (const [key, entryValue] of Object.entries(value)) {
    if (isNoisyResultKey(key)) {
      continue;
    }
    if (typeof entryValue === "string" || typeof entryValue === "number" || typeof entryValue === "boolean") {
      if (isAnswerLikeKey(key)) {
        pushAnswerCandidate(candidates, formatInlineValue(entryValue), { ...context, key });
      }
      continue;
    }
    collectAnswerCandidates(candidates, entryValue, { ...context, key }, depth + 1);
  }
}

function pushAnswerCandidate(candidates, text, context = {}) {
  const normalizedText = normalizeAnswerCandidateText(text);
  if (!normalizedText) {
    return;
  }
  candidates.push({
    text: normalizedText,
    score: scoreAnswerCandidate(normalizedText, context),
    source: context.source || "unknown",
    key: context.key || "",
  });
}

function selectBestAnswerCandidate(candidates) {
  if (!Array.isArray(candidates) || !candidates.length) {
    return null;
  }
  return candidates
    .slice()
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      if (left.text.length !== right.text.length) {
        return left.text.length - right.text.length;
      }
      return left.text.localeCompare(right.text);
    })[0] || null;
}

function isConcreteAnswerCandidate(candidate) {
  return Boolean(candidate && candidate.text && candidate.score >= 30);
}

function normalizeAnswerCandidateText(text) {
  const normalized = String(text || "")
    .split(/\n+/)
    .map((line) => String(line || "").trim())
    .filter(Boolean)
    .join("\n");
  return normalized;
}

function collectConcreteEquationLines(equations) {
  if (!Array.isArray(equations)) {
    return [];
  }
  return equations
    .map((equation) => normalizeAnswerCandidateText(formatInlineValue(equation)))
    .filter(Boolean)
    .filter((line) => !isMetaEquationText(line));
}

function isMetaEquationText(text) {
  const normalized = String(text || "").trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return RESULT_META_EQUATION_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

function isMetaAnswerText(text) {
  const normalized = String(text || "").trim();
  if (!normalized) {
    return false;
  }
  const lines = normalized.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length) {
    return false;
  }
  return lines.every((line) => isMetaEquationText(line) || RESULT_META_TEXT_PATTERNS.some((pattern) => pattern.test(line)));
}

function isAnswerLikeKey(key) {
  const normalized = String(key || "").trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return RESULT_ANSWER_KEYS.includes(normalized)
    || /answer|result|solution|output|fraction|probab|value|final|closed_form|steady_state/.test(normalized);
}

function scoreAnswerCandidate(text, context = {}) {
  const normalized = normalizeAnswerCandidateText(text);
  if (!normalized) {
    return Number.NEGATIVE_INFINITY;
  }

  let score = 0;
  const source = String(context.source || "");
  const key = String(context.key || "").trim().toLowerCase();

  if (source === "equations") {
    score += 70;
  } else if (source === "known_vars") {
    score += 56;
  } else if (source === "quantities") {
    score += 42;
  } else if (source === "quantities-summary") {
    score += 28;
  } else if (source === "boundary_conditions") {
    score += 34;
  } else if (source === "boundary-summary") {
    score += 22;
  } else if (source === "thought_step") {
    score -= 18;
  }

  if (key) {
    const keyIndex = RESULT_ANSWER_KEYS.indexOf(key);
    if (keyIndex >= 0) {
      score += Math.max(10, 26 - keyIndex * 2);
    } else if (isAnswerLikeKey(key)) {
      score += 12;
    }
  }

  if (/\b\d+\s*\/\s*\d+\b/.test(normalized)) {
    score += 34;
  }
  if (/\b\d+(?:\.\d+)?\b/.test(normalized)) {
    score += 14;
  }
  if (/[=≈≃<>≤≥]/.test(normalized)) {
    score += 22;
  }
  if (/\b(?:probability|final answer|answer|result|therefore|equals|is)\b/i.test(normalized)) {
    score += 10;
  }
  if (/\b(?:kg|m\/s|m\s*s\^-?1|m\/s\^2|n|j|pa|w|hz|rad\/s|s|k|c|v|a|ohm|t)\b/i.test(normalized)) {
    score += 10;
  }

  const lines = normalized.split(/\n+/).filter(Boolean);
  if (lines.length <= 3) {
    score += 6;
  }
  score -= Math.min(24, Math.floor(normalized.length / 140));

  if (isMetaAnswerText(normalized)) {
    score -= 140;
  }

  return score;
}

function formatStructuredAnswer(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => formatInlineValue(entry)).filter(Boolean).join("\n");
  }
  if (isPlainObject(value)) {
    return Object.entries(value)
      .map(([key, entryValue]) => `${key}: ${formatInlineValue(entryValue)}`)
      .filter((line) => !line.endsWith(": "))
      .join("\n");
  }
  return formatInlineValue(value);
}

function isNoisyResultKey(key) {
  return [
    "orchestrator_task",
    "hard_rule_check",
    "ignored_review_rule_violations",
    "ignored_violations",
    "local_model_fallback",
    "route_options",
    "step_blueprints",
  ].includes(String(key || ""));
}

function renderTree() {
  const frontierIds = new Set(
    Array.isArray(uiState.snapshot && uiState.snapshot.frontier)
      ? uiState.snapshot.frontier.map((entry) => entry.node_id)
      : []
  );

  if (!uiState.treeModel.root) {
    const empty = document.createElement("div");
    empty.className = "tree-line empty-line";
    empty.textContent = "No tree loaded. Create a session or attach to an existing session id.";
    dom.treeLines.replaceChildren(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  let selectedElement = null;

  uiState.treeModel.visibleNodes.forEach((entry, index) => {
    const line = document.createElement("button");
    line.type = "button";
    line.className = "tree-line";
    line.style.setProperty("--tree-depth", String(entry.depth));
    if (entry.depth === 0) {
      line.classList.add("tree-line-root");
    }
    if (entry.node.id === uiState.selectedNodeId) {
      line.classList.add("selected");
      selectedElement = line;
    }
    if (frontierIds.has(entry.node.id)) {
      line.classList.add("frontier");
    }
    if (uiState.searchMatches.includes(entry.node.id)) {
      line.classList.add("match");
    }
    line.dataset.nodeId = entry.node.id;
    line.setAttribute("aria-selected", entry.node.id === uiState.selectedNodeId ? "true" : "false");

    const contentSpan = document.createElement("span");
    contentSpan.className = "tree-line-content";

    const headerSpan = document.createElement("span");
    headerSpan.className = "tree-line-header";

    const glyphSpan = document.createElement("span");
    glyphSpan.className = "tree-line-fold";
    glyphSpan.textContent = entry.summary.foldMark;

    const titleSpan = document.createElement("span");
    titleSpan.className = "tree-line-title";
    titleSpan.textContent = entry.summary.title;

    const metaSpan = document.createElement("span");
    metaSpan.className = "tree-line-meta";
    metaSpan.textContent = `${String(index + 1).padStart(3, "0")} · ${entry.summary.meta}`;

    const chipRow = document.createElement("span");
    chipRow.className = "tree-line-chips";

    const statusBadge = document.createElement("span");
    statusBadge.className = `tree-badge tree-badge-status tree-badge-status-${entry.summary.statusTone}`;
    statusBadge.textContent = entry.summary.status;
    chipRow.append(statusBadge);

    if (entry.summary.routeFamily) {
      const routeBadge = document.createElement("span");
      routeBadge.className = "tree-badge tree-badge-route";
      routeBadge.textContent = entry.summary.routeFamily;
      chipRow.append(routeBadge);
    }

    if (entry.summary.stepFocus) {
      const stepBadge = document.createElement("span");
      stepBadge.className = "tree-badge tree-badge-step";
      stepBadge.textContent = entry.summary.stepFocus;
      chipRow.append(stepBadge);
    }

    if (entry.summary.ignoredNoiseCount) {
      const noiseBadge = document.createElement("span");
      noiseBadge.className = "tree-badge tree-badge-noise";
      noiseBadge.textContent = `noise ${entry.summary.ignoredNoiseCount}`;
      chipRow.append(noiseBadge);
    }

    if (frontierIds.has(entry.node.id)) {
      const frontierBadge = document.createElement("span");
      frontierBadge.className = "tree-badge tree-badge-frontier";
      frontierBadge.textContent = "frontier";
      chipRow.append(frontierBadge);
    }

    if (uiState.searchMatches.includes(entry.node.id)) {
      const matchBadge = document.createElement("span");
      matchBadge.className = "tree-badge tree-badge-match";
      matchBadge.textContent = "match";
      chipRow.append(matchBadge);
    }

    headerSpan.append(glyphSpan, titleSpan);
    contentSpan.append(headerSpan, metaSpan);

    line.append(contentSpan, chipRow);
    fragment.append(line);
  });

  dom.treeLines.replaceChildren(fragment);

  if (uiState.revealSelection && selectedElement) {
    selectedElement.scrollIntoView({ block: "nearest", inline: "nearest" });
    uiState.revealSelection = false;
  }
}

function renderDetailPane() {
  const selectedNode = getSelectedNode();
  const metaTask = uiState.snapshot && typeof uiState.snapshot.meta_task === "object"
    ? uiState.snapshot.meta_task
    : {};
  const runState = getRunState(uiState.snapshot);
  if (!selectedNode) {
    const fragment = document.createDocumentFragment();
    dom.detailSummary.textContent = Object.keys(metaTask).length > 0
      ? "session meta task | no node selected"
      : "no node selected";

    if (Object.keys(metaTask).length > 0) {
      fragment.append(createDetailSection("Session meta task", createDetailPairs(metaTask)));
    }

    fragment.append(
      createDetailSection(
        "Selection",
        createDetailTextBlock("Attach a session or select a node to inspect the current tree.")
      )
    );
    dom.detailBody.replaceChildren(fragment);
    return;
  }

  const selectedEntry = uiState.treeModel.allNodes.find((entry) => entry.node.id === selectedNode.id);
  const depth = selectedEntry ? selectedEntry.depth : 0;
  const childIds = getChildren(selectedNode).map((child) => child.id);
  const isSyntheticRoot = isSyntheticPlaceholderRoot(selectedNode);
  const fragment = document.createDocumentFragment();
  const visibleResult = displayResultState(selectedNode);
  const routeFamily = getNodeRouteFamily(selectedNode);
  const stepFocus = getNodeStepFocus(selectedNode);
  const ignoredNoiseCount = getNodeIgnoredNoiseCount(selectedNode);

  dom.detailSummary.textContent =
    `${selectedNode.id} | depth ${depth} | ${shortStatus(visibleResult)} | children ${childIds.length}${routeFamily ? ` | route ${routeFamily}` : ""}${stepFocus ? ` | step ${stepFocus}` : ""}${ignoredNoiseCount ? ` | noise ${ignoredNoiseCount}` : ""}`;

  fragment.append(
    createDetailFacts([
      ["Node", selectedNode.id],
      ["Parent", selectedNode.parent_id || "-"],
      ["Depth", String(depth)],
      ["Route", routeFamily || "-"],
      ["Step focus", stepFocus || "-"],
      ["Result", shortStatus(visibleResult)],
      ["Status", String(selectedNode.status || "-").toUpperCase()],
      ["Score", formatScore(selectedNode.score)],
      ["Ignored noise", ignoredNoiseCount ? String(ignoredNoiseCount) : "0"],
      ["Children", String(childIds.length)],
    ])
  );

  if (Object.keys(metaTask).length > 0) {
    fragment.append(createDetailSection("Session meta task", createDetailPairs(metaTask)));
  }

  if (isSyntheticRoot) {
    fragment.append(
      createDetailSection(
        "Session run state",
        createDetailFacts([
          ["Run status", String(runState.status || "-")],
          ["Run phase", String(runState.phase || "-")],
          ["Problem context prepared", String(Boolean(runState.problem_context_prepared))],
          ["Auto run requested", String(Boolean(runState.auto_run_requested))],
          ["In-flight expansion", formatInFlightExpansion(uiState.snapshot) || "-"],
          ["Last error", String(runState.last_error || "-")],
        ])
      )
    );
  }

  fragment.append(
    createDetailSection(
      "Thought step",
      createDetailTextBlock(formatInlineValue(selectedNode.thought_step) || "No thought step recorded.")
    ),
    createDetailSection("Equations", createDetailList(selectedNode.equations)),
    createDetailSection("Known vars", createDetailPairs(selectedNode.known_vars)),
    createDetailSection("Quantities", createDetailPairs(selectedNode.quantities)),
    createDetailSection("Boundary conditions", createDetailPairs(selectedNode.boundary_conditions)),
    createDetailSection("Used models", createDetailList(selectedNode.used_models)),
    createDetailSection("Reflection history", createDetailList(selectedNode.reflection_history)),
    createDetailSection("Child nodes", createDetailList(childIds)),
  );

  dom.detailBody.replaceChildren(fragment);
}

function renderDetailEmpty(message) {
  const empty = document.createElement("div");
  empty.className = "detail-empty";
  empty.textContent = message;
  dom.detailBody.replaceChildren(empty);
}

function createDetailFacts(items) {
  const grid = document.createElement("div");
  grid.className = "detail-facts";

  items.forEach(([label, value]) => {
    const fact = document.createElement("div");
    fact.className = "detail-fact";

    const labelSpan = document.createElement("div");
    labelSpan.className = "detail-fact-label";
    labelSpan.textContent = label;

    const valueSpan = document.createElement("div");
    valueSpan.className = "detail-fact-value";
    valueSpan.textContent = value;

    fact.append(labelSpan, valueSpan);
    grid.append(fact);
  });

  return grid;
}

function createDetailSection(title, contentNode) {
  const section = document.createElement("section");
  section.className = "detail-section";

  const titleNode = document.createElement("div");
  titleNode.className = "detail-section-title";
  titleNode.textContent = title;

  section.append(titleNode, contentNode);
  return section;
}

function createDetailTextBlock(text) {
  const block = document.createElement("div");
  block.className = "detail-text";
  block.textContent = text;
  return block;
}

function createDetailList(values) {
  const normalized = Array.isArray(values)
    ? values.map((value) => formatInlineValue(value)).filter(Boolean)
    : [];

  if (normalized.length === 0) {
    return createDetailEmptyBlock("None");
  }

  const list = document.createElement("ul");
  list.className = "detail-list";

  normalized.forEach((value) => {
    const item = document.createElement("li");
    item.textContent = value;
    list.append(item);
  });

  return list;
}

function createDetailPairs(value) {
  const entries = normalizeDetailEntries(value);
  if (entries.length === 0) {
    return createDetailEmptyBlock("None");
  }

  const wrapper = document.createElement("div");
  wrapper.className = "detail-pairs";

  entries.forEach(([key, entryValue]) => {
    const row = document.createElement("div");
    row.className = "detail-pair";

    const keyNode = document.createElement("div");
    keyNode.className = "detail-pair-key";
    keyNode.textContent = key;

    const valueNode = document.createElement("div");
    valueNode.className = "detail-pair-value";
    valueNode.textContent = entryValue;

    row.append(keyNode, valueNode);
    wrapper.append(row);
  });

  return wrapper;
}

function createDetailEmptyBlock(text) {
  const empty = document.createElement("div");
  empty.className = "detail-empty";
  empty.textContent = text;
  return empty;
}

function normalizeDetailEntries(value) {
  if (Array.isArray(value)) {
    return value
      .map((entry, index) => [String(index + 1), formatInlineValue(entry)])
      .filter(([, entryValue]) => Boolean(entryValue));
  }
  if (value && typeof value === "object") {
    return Object.entries(value)
      .map(([key, entryValue]) => [key, formatInlineValue(entryValue)])
      .filter(([, entryValue]) => Boolean(entryValue));
  }

  const text = formatInlineValue(value);
  return text ? [["value", text]] : [];
}

function formatInlineValue(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value);
  }
}

function renderFrontier() {
  const frontier = Array.isArray(uiState.snapshot && uiState.snapshot.frontier)
    ? uiState.snapshot.frontier
    : [];
  const inFlightDetail = formatInFlightExpansion(uiState.snapshot);

  if (frontier.length === 0) {
    dom.frontierList.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "meta-line";
    empty.textContent = inFlightDetail && isSessionBusy(uiState.snapshot)
      ? `frontier temporarily empty | ${inFlightDetail}`
      : "frontier empty";
    dom.frontierList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  frontier.forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "frontier-item";
    button.dataset.nodeId = entry.node_id;

    const idLine = document.createElement("div");
    idLine.className = "frontier-item-id";
    idLine.textContent = entry.node_id;

    const metaLine = document.createElement("div");
    metaLine.className = "frontier-item-meta";
    metaLine.textContent =
      `depth ${entry.depth} | priority ${formatScore(entry.priority)} | score ${formatScore(entry.score)} | ${entry.status}`;

    button.append(idLine, metaLine);
    fragment.append(button);
  });

  dom.frontierList.replaceChildren(fragment);
}

function renderActivity() {
  const entries = Array.isArray(uiState.snapshot && uiState.snapshot.expansion_log)
    ? uiState.snapshot.expansion_log
    : [];

  if (entries.length === 0) {
    dom.activityLog.textContent = "No events yet.";
    return;
  }

  const formatted = entries
    .slice(-8)
    .reverse()
    .map((entry, index) => formatActivityEntry(entry, entries.length - index))
    .join("\n\n");
  dom.activityLog.textContent = formatted;
}

function formatActivityEntry(entry, ordinal) {
  const header = `${String(ordinal).padStart(2, "0")} ${String(entry.event || "event").toUpperCase()}`;
  const details = [];
  if (entry.node_id) {
    details.push(`node=${entry.node_id}`);
  }
  if (Array.isArray(entry.deleted_node_ids) && entry.deleted_node_ids.length) {
    details.push(`deleted=${entry.deleted_node_ids.join(",")}`);
  }
  if (entry.frontier_size_after !== undefined) {
    details.push(`frontier=${entry.frontier_size_after}`);
  }
  return `${header}\n${details.join(" | ")}\n${trimText(JSON.stringify(entry), 300)}`;
}

function selectNode(nodeId) {
  if (!nodeId || !uiState.treeModel.nodeById.has(nodeId)) {
    return;
  }
  uiState.selectedNodeId = nodeId;
  uiState.revealSelection = true;
  render();
}

function moveSelection(direction) {
  if (!uiState.treeModel.visibleNodes.length) {
    return;
  }
  const currentIndex = uiState.treeModel.visibleNodes.findIndex(
    (entry) => entry.node.id === uiState.selectedNodeId,
  );
  const startIndex = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = clamp(startIndex + direction, 0, uiState.treeModel.visibleNodes.length - 1);
  selectNode(uiState.treeModel.visibleNodes[nextIndex].node.id);
}

function navigateLeft() {
  const node = getSelectedNode();
  if (!node) {
    return;
  }
  const children = getChildren(node);
  if (children.length > 0 && !uiState.collapsedNodeIds.has(node.id)) {
    uiState.collapsedNodeIds.add(node.id);
    setStatus(`Collapsed ${node.id}.`, "ok");
    render();
    return;
  }
  if (node.parent_id) {
    selectNode(node.parent_id);
  }
}

function navigateRight() {
  const node = getSelectedNode();
  if (!node) {
    return;
  }
  const children = getChildren(node);
  if (children.length === 0) {
    return;
  }
  if (uiState.collapsedNodeIds.has(node.id)) {
    uiState.collapsedNodeIds.delete(node.id);
    uiState.revealSelection = true;
    setStatus(`Expanded ${node.id}.`, "ok");
    render();
    return;
  }
  selectNode(children[0].id);
}

function toggleSelectedNodeExpansion(nodeId) {
  const node = uiState.treeModel.nodeById.get(nodeId);
  if (!node || getChildren(node).length === 0) {
    return;
  }
  if (uiState.collapsedNodeIds.has(node.id)) {
    uiState.collapsedNodeIds.delete(node.id);
    setStatus(`Expanded ${node.id}.`, "ok");
  } else {
    uiState.collapsedNodeIds.add(node.id);
    setStatus(`Collapsed ${node.id}.`, "ok");
  }
  uiState.revealSelection = true;
  render();
}

function collapseAllDescendants() {
  if (!uiState.treeModel.root) {
    return;
  }
  const descendantsWithChildren = uiState.treeModel.allNodes
    .filter((entry) => entry.depth > 0 && entry.childCount > 0)
    .map((entry) => entry.node.id);
  uiState.collapsedNodeIds = new Set(descendantsWithChildren);
  uiState.revealSelection = true;
  setStatus("Collapsed all descendant branches.", "ok");
  render();
}

function jumpToSearchMatch(direction) {
  if (uiState.searchMatches.length === 0) {
    setStatus("No search matches.", "error");
    return;
  }

  const currentIndex = uiState.searchMatches.indexOf(uiState.selectedNodeId);
  const baseIndex = currentIndex >= 0 ? currentIndex : uiState.searchCursor;
  const nextIndex = modulo(baseIndex + direction, uiState.searchMatches.length);
  const targetNodeId = uiState.searchMatches[nextIndex];
  revealNode(targetNodeId);
  uiState.searchCursor = nextIndex;
  selectNode(targetNodeId);
  setStatus(`Jumped to match ${nextIndex + 1}/${uiState.searchMatches.length}.`, "ok");
}

function revealNode(nodeId) {
  const ancestors = uiState.treeModel.ancestorsById.get(nodeId) || [];
  ancestors.forEach((ancestorId) => uiState.collapsedNodeIds.delete(ancestorId));
}

function panViewport(key) {
  const stepY = 48;
  const stepX = 96;
  switch (key) {
    case "ArrowUp":
      dom.treeViewport.scrollBy({ top: -stepY, behavior: "smooth" });
      break;
    case "ArrowDown":
      dom.treeViewport.scrollBy({ top: stepY, behavior: "smooth" });
      break;
    case "ArrowLeft":
      dom.treeViewport.scrollBy({ left: -stepX, behavior: "smooth" });
      break;
    case "ArrowRight":
      dom.treeViewport.scrollBy({ left: stepX, behavior: "smooth" });
      break;
    default:
      break;
  }
}

function getSelectedNode() {
  return uiState.selectedNodeId ? uiState.treeModel.nodeById.get(uiState.selectedNodeId) || null : null;
}

async function createSession() {
  if (uiState.requestInFlight) {
    return;
  }

  const payload = buildCreateSessionPayload();
  await withRequest("Creating session and starting run...", async () => {
    const response = await apiRequest("/api/tot/sessions", {
      method: "POST",
      body: payload,
    });
    applySessionState(
      response.session_id,
      response.state,
      "Session created.",
      isSessionBusy(response.state) ? "busy" : "ok",
      summarizeSessionAction("create", null, response.state),
    );
    enableAutoRefreshForSession();
  });
}

async function attachSession(options = {}) {
  const sessionId = dom.sessionIdInput.value.trim();
  if (!sessionId) {
    setStatus("Session id is required to attach.", "error");
    return;
  }

  await withRequest(options.silent ? null : "Attaching session...", async () => {
    const response = await apiRequest(`/api/tot/sessions/${encodeURIComponent(sessionId)}`);
    applySessionState(
      response.session_id,
      response.state,
      options.silent ? "Session restored." : "Session attached.",
      "ok",
      summarizeSessionAction("attach", null, response.state),
    );
    enableAutoRefreshForSession();
  }, options);
}

async function refreshSession(options = {}) {
  if (!uiState.sessionId) {
    if (!options.silent) {
      setStatus("No session attached.", "error");
    }
    return;
  }

  await withRequest(options.silent ? null : "Refreshing session...", async () => {
    const previousSnapshot = uiState.snapshot;
    const response = await apiRequest(`/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}`);
    const refreshSummary = summarizeSessionAction("refresh", previousSnapshot, response.state);
    const hasVisibleChange = hasVisibleMetricsChange(
      getSnapshotMetrics(previousSnapshot),
      getSnapshotMetrics(response.state)
    );
    const hasStableIdleSnapshot = Boolean(
      !isSessionBusy(response.state)
      && !hasVisibleChange
    );
    const matchesCurrentFeedback = Boolean(
      refreshSummary.title === uiState.lastActionTitle
      && refreshSummary.detail === uiState.lastActionDetail
      && (refreshSummary.tone || "ok") === uiState.lastActionTone
    );
    const preserveSilentFeedback = Boolean(
      options.silent
      && (hasStableIdleSnapshot || (!hasVisibleChange && matchesCurrentFeedback))
    );
    if (options.silent && hasStableIdleSnapshot && uiState.pollingEnabled) {
      uiState.pollingEnabled = false;
      dom.pollingToggle.checked = false;
      restartPolling();
    }
    applySessionState(
      response.session_id,
      response.state,
      preserveSilentFeedback
        ? uiState.statusMessage
        : options.silent ? refreshSummary.statusMessageSilent : refreshSummary.statusMessage,
      preserveSilentFeedback ? uiState.statusTone : refreshSummary.tone,
      preserveSilentFeedback
        ? {
            title: uiState.lastActionTitle,
            detail: uiState.lastActionDetail,
            tone: uiState.lastActionTone,
            record: false,
          }
        : refreshSummary,
      { preserveUpdatedAt: preserveSilentFeedback },
    );
  }, options);
}

async function runSession() {
  if (!uiState.sessionId) {
    setStatus("Create or attach a session first.", "error");
    return;
  }

  await withRequest("Running scheduler...", async () => {
    const previousSnapshot = uiState.snapshot;
    const response = await apiRequest(`/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}/run`, {
      method: "POST",
    });
    const runSummary = summarizeRunAction(previousSnapshot, response.state);
    applySessionState(
      response.session_id,
      response.state,
      runSummary.statusMessage,
      runSummary.tone,
      runSummary,
    );
  });
}

async function deleteSelectedNode(options = {}) {
  if (!uiState.sessionId) {
    setStatus("Create or attach a session first.", "error");
    return;
  }
  const selectedNode = getSelectedNode();
  if (!selectedNode) {
    setStatus("Select a node before deleting.", "error");
    return;
  }
  if (!selectedNode.parent_id) {
    setStatus("Root deletion is blocked by the API.", "error");
    return;
  }

  const reason = dom.deleteReasonInput.value.trim();
  if (!reason) {
    setStatus("Deletion reason is required for backend review.", "error");
    return;
  }
  const steerPrompt = dom.steerPromptInput.value.trim();
  const shouldSteer = Boolean(options.steer);
  if (options.steer && !steerPrompt) {
    dom.steerPromptInput.focus();
    setStatus("Steer prompt is required for delete + steer.", "error");
    return;
  }

  const requestLabel = shouldSteer
    ? `Deleting ${selectedNode.id}, applying steer prompt, then continuing...`
    : `Submitting delete review for ${selectedNode.id}...`;

  await withRequest(requestLabel, async () => {
    const response = await apiRequest(
      `/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}/nodes/${encodeURIComponent(selectedNode.id)}`,
      {
        method: "DELETE",
        body: {
          reason,
          requested_by: "frontend-terminal-gui",
          steer_prompt: shouldSteer ? steerPrompt : "",
          run_after_delete: Boolean(options.runAfterDelete),
        },
      },
    );
    uiState.selectedNodeId = selectedNode.parent_id;
    uiState.revealSelection = true;
    applySessionState(
      response.session_id,
      response.state,
      response.deleted
        ? `Deleted ${response.deleted_node_ids.length} node(s) after review.`
        : `Delete rejected: ${response.review && response.review.reason ? response.review.reason : "not approved"}`,
      response.deleted ? "ok" : "error",
      response.deleted
        ? {
            title: shouldSteer
              ? `Delete approved and steer applied for ${selectedNode.id}.`
              : `Delete approved for ${selectedNode.id}.`,
            detail: buildDeleteActionDetail(response, shouldSteer),
            tone: "ok",
          }
        : {
            title: `Delete rejected for ${selectedNode.id}.`,
            detail: response.review && response.review.reason
              ? String(response.review.reason)
              : "The backend review model did not approve the deletion.",
            tone: "error",
          },
    );
  });
}

function buildDeleteActionDetail(response, shouldSteer) {
  const parts = [`Removed ${response.deleted_node_ids.length} node(s).`];
  const steering = response.steering && typeof response.steering === "object" ? response.steering : {};
  if (shouldSteer) {
    parts.push(
      steering.applied
        ? `Steer queued on parent ${steering.parent_id || response.parent_id || "unknown"}.`
        : `Steer not queued${steering.reason ? `: ${steering.reason}` : "."}`,
    );
  }
  parts.push(formatSnapshotSummary(response.state));
  return parts.join(" ");
}

async function dropSession() {
  if (!uiState.sessionId) {
    setStatus("No session attached.", "error");
    return;
  }
  const confirmed = window.confirm(`Drop session ${uiState.sessionId}?`);
  if (!confirmed) {
    return;
  }

  await withRequest("Dropping session...", async () => {
    const droppedSessionId = uiState.sessionId;
    await apiRequest(`/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}`, {
      method: "DELETE",
    });
    clearSessionState();
    setStatus("Session dropped.", "ok");
    setActionFeedback(
      `Session ${droppedSessionId} dropped.`,
      "The backend session was deleted and the local tree view was cleared.",
      "ok",
    );
  });
}

function buildCreateSessionPayload() {
  const problemContext = parseJsonText(dom.problemContextInput.value, "problem context", {});
  const scheduler = parseJsonText(dom.schedulerConfigInput.value, "scheduler config", DEFAULT_SCHEDULER_CONFIG);
  const problemStatement = dom.problemPromptInput.value.trim();
  if (!isPlainObject(problemContext)) {
    throw new Error("Problem context JSON must decode to an object.");
  }
  if (!isPlainObject(scheduler)) {
    throw new Error("Scheduler config JSON must decode to an object.");
  }
  if (!problemStatement) {
    dom.problemPromptInput.focus();
    throw new Error("Problem to solve is required before creating a session.");
  }

  const backend = {
    base_url: dom.baseUrlInput.value.trim(),
    planning_model: readModelInput(dom.planningModelInput, DEFAULT_PLANNING_MODEL),
    modeling_model: readModelInput(dom.modelingModelInput, DEFAULT_MODELING_MODEL),
    review_model: readModelInput(dom.reviewModelInput, DEFAULT_REVIEW_MODEL),
    non_terminal_evaluation_model: readModelInput(
      dom.nonTerminalEvaluationModelInput,
      DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    ),
    timeout: parsePositiveNumber(dom.timeoutInput.value, "timeout"),
  };

  const depthPreset = normalizeDepthPreset(scheduler.depth_preset);

  return {
    problem_context: {
      ...problemContext,
      problem_statement: problemStatement,
      reasoning_depth_preset: depthPreset,
    },
    backend,
    scheduler: {
      ...scheduler,
      depth_preset: depthPreset,
    },
    run_on_create: true,
  };
}

function extractProblemStatementDraft(rawValue) {
  try {
    const parsed = JSON.parse(String(rawValue || "{}"));
    if (!isPlainObject(parsed)) {
      return "";
    }
    const statement = parsed.problem_statement;
    return typeof statement === "string" ? statement.trim() : "";
  } catch (_error) {
    return "";
  }
}

function sanitizeModelName(value, fallback) {
  const normalized = typeof value === "string" ? value.trim() : "";
  return normalized || fallback;
}

function isLegacyQwopusModel(value) {
  const normalized = sanitizeModelName(value, "");
  if (!normalized.startsWith("mlx-")) {
    return false;
  }
  const suffix = normalized.slice(4);
  return suffix === LEGACY_QWOPUS_MODEL_FAMILY || suffix === `${LEGACY_QWOPUS_MODEL_FAMILY}:2`;
}

function migrateLegacyPlanningModel(value) {
  const normalized = sanitizeModelName(value, DEFAULT_PLANNING_MODEL);
  return normalized === PREVIOUS_DEFAULT_PLANNING_MODEL || isLegacyQwopusModel(normalized)
    ? DEFAULT_PLANNING_MODEL
    : normalized;
}

function migrateLegacyModelingModel(value) {
  const normalized = sanitizeModelName(value, DEFAULT_MODELING_MODEL);
  return isLegacyQwopusModel(normalized) || LEGACY_MODELING_DEFAULTS.includes(normalized)
    ? DEFAULT_MODELING_MODEL
    : normalized;
}

function migrateLegacyReviewModel(value) {
  const normalized = sanitizeModelName(value, DEFAULT_REVIEW_MODEL);
  return isLegacyQwopusModel(normalized) || LEGACY_REVIEW_DEFAULTS.includes(normalized)
    ? DEFAULT_REVIEW_MODEL
    : normalized;
}

function applySessionState(sessionId, snapshot, message, tone = "ok", actionFeedback = null, options = {}) {
  uiState.sessionId = String(sessionId || "");
  uiState.snapshot = snapshot || null;
  if (!options.preserveUpdatedAt) {
    uiState.lastUpdatedAt = new Date();
  }
  dom.sessionIdInput.value = uiState.sessionId;
  persistDraft();
  if (isSessionBusy(snapshot)) {
    scheduleBusyRefresh();
  } else {
    clearBusyRefresh();
  }
  setStatus(message, tone, { record: !actionFeedback });
  if (actionFeedback) {
    setActionFeedback(actionFeedback.title, actionFeedback.detail, actionFeedback.tone || tone, {
      record: actionFeedback.record !== false,
    });
  }
  render();
}

function clearSessionState() {
  clearBusyRefresh();
  uiState.sessionId = "";
  uiState.snapshot = null;
  uiState.selectedNodeId = null;
  uiState.collapsedNodeIds.clear();
  uiState.searchMatches = [];
  uiState.lastUpdatedAt = null;
  dom.sessionIdInput.value = "";
  persistDraft();
  render();
}

async function withRequest(message, operation, options = {}) {
  if (uiState.requestInFlight) {
    if (!options.silent) {
      setStatus("Another request is still running.", "error");
    }
    return;
  }

  uiState.requestInFlight = true;
  if (message) {
    setStatus(message, "busy", { record: false });
    setActionFeedback(message, "Waiting for the backend response...", "busy");
  }
  renderButtons();

  try {
    await operation();
  } catch (error) {
    if (options.silent && error instanceof Error && /Session not found/i.test(error.message)) {
      uiState.pollingEnabled = false;
      dom.pollingToggle.checked = false;
      restartPolling();
      clearSessionState();
      setStatus("Saved session expired. Start a new session.", "idle", { record: false });
      setActionFeedback(
        "Saved session expired.",
        "Auto refresh stopped because the backend no longer recognizes the saved session id.",
        "warn",
      );
      return;
    }
    handleError(error);
  } finally {
    uiState.requestInFlight = false;
    render();
  }
}

async function apiRequest(path, options = {}) {
  const request = {
    method: options.method || "GET",
    headers: {},
  };
  if (options.body !== undefined) {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(options.body);
  }

  const response = await window.fetch(path, request);
  const text = await response.text();
  const payload = parseResponsePayload(text);

  if (!response.ok) {
    if (payload && typeof payload === "object" && payload.detail) {
      throw new Error(String(payload.detail));
    }
    throw new Error(`HTTP ${response.status}`);
  }

  return payload;
}

function parseResponsePayload(text) {
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (_error) {
    return text;
  }
}

function handleError(error) {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(message, "error", { record: false });
  setActionFeedback("Request failed.", message, "error");
}

function summarizeSessionAction(action, previousSnapshot, nextSnapshot) {
  const nextMetrics = getSnapshotMetrics(nextSnapshot);
  const previousMetrics = getSnapshotMetrics(previousSnapshot);
  const deltas = getSnapshotDeltas(previousMetrics, nextMetrics);
  const runState = getRunState(nextSnapshot);
  const sessionBusy = isSessionBusy(nextSnapshot);

  if (action === "create") {
    return {
      title: sessionBusy
        ? `Session created: ${nextMetrics.sessionHint}.`
        : `Session ready: ${nextMetrics.rootId || "new root"}.`,
      detail: sessionBusy
        ? `Session code issued before meta-task preparation. Background phase: ${runState.phase || "queued"}. ${formatSnapshotSummary(nextSnapshot)}`.trim()
        : `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim(),
      tone: sessionBusy ? "busy" : "ok",
      statusMessage: sessionBusy ? "Session created. Background initialization started." : "Session created.",
      statusMessageSilent: sessionBusy ? "Background initialization started." : "Session created.",
    };
  }

  if (action === "attach") {
    return {
      title: `Loaded session ${nextMetrics.sessionHint}.`,
      detail: sessionBusy
        ? `Background phase: ${runState.phase || "busy"}. ${formatSnapshotSummary(nextSnapshot)}`
        : `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim(),
      tone: sessionBusy ? "busy" : "ok",
      statusMessage: sessionBusy ? "Session attached. Background run still active." : "Session attached.",
      statusMessageSilent: sessionBusy ? "Background run still active." : "Session restored.",
    };
  }

  const changed = hasVisibleMetricsChange(previousMetrics, nextMetrics);
  if (sessionBusy) {
    return {
      title: changed ? "Background run advanced the tree." : "Background run still in progress.",
      detail: `${formatSnapshotSummary(nextSnapshot)} phase ${runState.phase || "busy"}. ${formatDeltaSummary(deltas)}`.trim(),
      tone: "busy",
      statusMessage: changed ? "Background run advanced the tree." : "Background run still in progress.",
      statusMessageSilent: changed ? "Background run advanced the tree." : "Background run still in progress.",
    };
  }

  return {
    title: changed ? "Session state refreshed." : "Refresh succeeded with no tree change.",
    detail: changed
      ? `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim()
      : `${formatSnapshotSummary(nextSnapshot)} No nodes, frontier entries, or expansion counters changed since the previous snapshot.`,
    tone: changed ? "ok" : "warn",
    statusMessage: changed ? "Session refreshed." : "Refresh completed with no visible change.",
    statusMessageSilent: changed ? "Auto refreshed." : "Auto refresh found no change.",
  };
}

function runUiAction(action) {
  Promise.resolve()
    .then(action)
    .catch((error) => {
      handleError(error);
      render();
    });
}

function exportCurrentAnswer() {
  const board = buildResultsBoard(uiState.treeModel);
  if (!shouldShowResultsBoard(uiState.snapshot, board) || !board.best) {
    setStatus("No final answer is ready to export yet.", "error");
    return;
  }

  const best = board.best;
  const safeSessionId = (uiState.sessionId || "session")
    .replace(/[^a-z0-9_-]+/gi, "-")
    .replace(/^-+|-+$/g, "") || "session";
  const fileName = `${safeSessionId}-final-answer.txt`;
  const fileText = [
    "ToT Final Answer Export",
    `Session: ${uiState.sessionId || "n/a"}`,
    `Node: ${best.node.id}`,
    `Score: ${formatScore(best.node.score)}`,
    `Summary: ${dom.resultSummary.textContent || ""}`,
    "",
    "Final Answer",
    buildNodeAnswerText(best.node, { maxLength: 20_000 }),
    "",
    "Winning Branch",
    summarizeResultThought(best.node),
    "",
    "Metadata",
    buildResultMeta(best),
  ].join("\n");

  const blob = new Blob([fileText], { type: "text/plain;charset=utf-8" });
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);

  setStatus("Final answer exported.", "ok");
  setActionFeedback(
    "Answer exported.",
    `Downloaded ${fileName} from ${best.node.id}.`,
    "ok",
  );
}

function summarizeRunAction(previousSnapshot, nextSnapshot) {
  const previousMetrics = getSnapshotMetrics(previousSnapshot);
  const nextMetrics = getSnapshotMetrics(nextSnapshot);
  const deltas = getSnapshotDeltas(previousMetrics, nextMetrics);

  if (deltas.expansions > 0 || deltas.nodes > 0) {
    return {
      title: "Run advanced the tree.",
      detail: `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim(),
      tone: "ok",
      statusMessage: `Run completed: +${Math.max(0, deltas.expansions)} expansion(s), +${Math.max(0, deltas.nodes)} node(s).`,
    };
  }

  const stallReason = inferRunStallReason(nextMetrics);
  return {
    title: stallReason.title,
    detail: `${stallReason.detail} ${formatSnapshotSummary(nextSnapshot)}`.trim(),
    tone: "warn",
    statusMessage: stallReason.statusMessage,
  };
}

function inferRunStallReason(metrics) {
  if (metrics.frontier === 0) {
    return {
      title: "Run stopped: no expandable frontier remained.",
      detail: "There were no frontier nodes left that the scheduler could expand.",
      statusMessage: "Run stopped: frontier empty.",
    };
  }
  return {
    title: "Run completed with no visible tree change.",
    detail: "The backend returned successfully, but the snapshot counters did not change. This usually means the scheduler revisited state without finding an expandable branch.",
    statusMessage: "Run completed with no visible change.",
  };
}

function getSnapshotMetrics(snapshot) {
  const root = snapshot && snapshot.root ? snapshot.root : null;
  const frontier = Array.isArray(snapshot && snapshot.frontier) ? snapshot.frontier.length : 0;
  const expansionLog = Array.isArray(snapshot && snapshot.expansion_log) ? snapshot.expansion_log : [];
  const expansionsUsed = Number.isFinite(snapshot && snapshot.expansions_used)
    ? Number(snapshot.expansions_used)
    : 0;
  const allNodes = root ? flattenNodes(root) : [];
  const runState = getRunState(snapshot);
  const inFlight = getInFlightExpansionMetrics(snapshot);

  return {
    rootId: root && root.id ? String(root.id) : "",
    sessionHint: uiState.sessionId || dom.sessionIdInput.value.trim() || "current session",
    nodes: allNodes.length,
    active: allNodes.filter((node) => String(node.status || "") === "ACTIVE").length,
    frontier,
    expansionsUsed,
    runStatus: String(runState.status || "idle"),
    runPhase: String(runState.phase || "created"),
    inFlightParentId: inFlight ? inFlight.parentId : "",
    inFlightBuiltChildCount: inFlight ? inFlight.builtChildCount : 0,
    inFlightExpectedChildCount: inFlight ? inFlight.expectedChildCount : 0,
    lastEvent: expansionLog.length ? expansionLog[expansionLog.length - 1] : null,
  };
}

function getSnapshotDeltas(previousMetrics, nextMetrics) {
  return {
    nodes: nextMetrics.nodes - previousMetrics.nodes,
    frontier: nextMetrics.frontier - previousMetrics.frontier,
    expansions: nextMetrics.expansionsUsed - previousMetrics.expansionsUsed,
  };
}

function hasVisibleMetricsChange(previousMetrics, nextMetrics) {
  const deltas = getSnapshotDeltas(previousMetrics, nextMetrics);
  return (
    deltas.nodes !== 0
    || deltas.frontier !== 0
    || deltas.expansions !== 0
    || previousMetrics.runStatus !== nextMetrics.runStatus
    || previousMetrics.runPhase !== nextMetrics.runPhase
    || previousMetrics.inFlightParentId !== nextMetrics.inFlightParentId
    || previousMetrics.inFlightBuiltChildCount !== nextMetrics.inFlightBuiltChildCount
    || previousMetrics.inFlightExpectedChildCount !== nextMetrics.inFlightExpectedChildCount
  );
}

function formatSnapshotSummary(snapshot) {
  const metrics = getSnapshotMetrics(snapshot);
  const runState = metrics.runStatus && metrics.runStatus !== "idle"
    ? ` | ${metrics.runStatus.toLowerCase()} ${metrics.runPhase.toLowerCase()}`
    : "";
  const inFlight = metrics.inFlightExpectedChildCount > 0
    ? ` | in-flight ${metrics.inFlightBuiltChildCount}/${metrics.inFlightExpectedChildCount}`
    : "";
  return `nodes ${metrics.nodes} | active ${metrics.active} | frontier ${metrics.frontier} | expansions ${metrics.expansionsUsed}${runState}${inFlight}`;
}

function formatDeltaSummary(deltas) {
  const parts = [];
  if (deltas.nodes !== 0) {
    parts.push(`${signedCount(deltas.nodes)} node(s)`);
  }
  if (deltas.frontier !== 0) {
    parts.push(`${signedCount(deltas.frontier)} frontier`);
  }
  if (deltas.expansions !== 0) {
    parts.push(`${signedCount(deltas.expansions)} expansion(s)`);
  }
  return parts.length ? `Change: ${parts.join(" | ")}.` : "";
}

function signedCount(value) {
  return value > 0 ? `+${value}` : String(value);
}

function shortUiTone(tone) {
  const normalized = String(tone || "idle").toUpperCase();
  if (normalized === "OK") {
    return "OK";
  }
  if (normalized === "ERROR") {
    return "ERROR";
  }
  if (normalized === "WARN") {
    return "WARN";
  }
  if (normalized === "BUSY") {
    return "BUSY";
  }
  return "INFO";
}

function flattenNodes(root) {
  if (!root) {
    return [];
  }
  const nodes = [];
  const stack = [root];
  while (stack.length) {
    const current = stack.pop();
    if (!current || typeof current !== "object") {
      continue;
    }
    nodes.push(current);
    const children = Array.isArray(current.children) ? current.children : [];
    for (let index = children.length - 1; index >= 0; index -= 1) {
      stack.push(children[index]);
    }
  }
  return nodes;
}

function parseJsonText(rawValue, label, fallbackValue) {
  const text = String(rawValue || "").trim();
  if (!text) {
    return fallbackValue;
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} JSON is invalid: ${error instanceof Error ? error.message : error}`);
  }
}

function parsePositiveNumber(rawValue, label) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive number.`);
  }
  return parsed;
}

function shortStatus(status) {
  const normalized = String(status || "UNKNOWN").toUpperCase();
  if (normalized === "PASS") {
    return "PASS";
  }
  if (normalized === "DROP") {
    return "DROP";
  }
  if (normalized === "FINALIZE") {
    return "FINAL";
  }
  if (normalized === "ACTIVE") {
    return "ACT";
  }
  if (normalized === "PRUNED_BY_RULE") {
    return "RULE";
  }
  if (normalized === "PRUNED_BY_SLM") {
    return "SLM";
  }
  return normalized;
}

function statusTone(status) {
  const normalized = String(status || "UNKNOWN").toUpperCase();
  if (normalized === "PASS") {
    return "active";
  }
  if (normalized === "DROP") {
    return "rule";
  }
  if (normalized === "FINALIZE") {
    return "solved";
  }
  if (normalized === "ACTIVE") {
    return "active";
  }
  if (normalized === "PRUNED_BY_RULE") {
    return "rule";
  }
  if (normalized === "PRUNED_BY_SLM") {
    return "slm";
  }
  if (normalized === "SOLVED") {
    return "solved";
  }
  return "unknown";
}

function formatScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return numeric.toFixed(2);
}

function trimText(text, maxLength) {
  const normalized = String(text || "");
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 3))}...`;
}

function safeJson(value) {
  try {
    return JSON.stringify(value || {});
  } catch (_error) {
    return String(value);
  }
}

function joinStrings(value) {
  return Array.isArray(value) ? value.join(" ") : "";
}

function displayResultState(node) {
  const explicit = String(node && node.result_state ? node.result_state : "").trim().toUpperCase();
  if (explicit) {
    return explicit;
  }
  const normalizedStatus = String(node && node.status ? node.status : "UNKNOWN").trim().toUpperCase();
  if (normalizedStatus === "SOLVED") {
    return "FINALIZE";
  }
  if (normalizedStatus.startsWith("PRUNED")) {
    return "DROP";
  }
  if (normalizedStatus === "ACTIVE") {
    return "PASS";
  }
  return normalizedStatus || "UNKNOWN";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function modulo(value, divisor) {
  return ((value % divisor) + divisor) % divisor;
}

function formatClock(date) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}