/* ==========================================================
   TCS Signal Check — script.js
   Talks to the FastAPI backend: GET /health, POST /predict
   ========================================================== */

"use strict";

const API_BASE_URL = "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 15000;

/* ----------------------------------------------------------
   1. Field definitions
   Mirrors the PredictionInput Pydantic model in main.py.
   rule: "gt0"  -> must be greater than 0
         "ge0"  -> must be 0 or more
         "pct"  -> 0 to 100 (RSI)
         "any"  -> any finite number
---------------------------------------------------------- */
const FIELD_GROUPS = [
  {
    title: "Price and volume",
    desc: "Today’s trading session for TCS.",
    fields: [
      { name: "Open",   label: "Open",   rule: "gt0" },
      { name: "High",   label: "High",   rule: "gt0" },
      { name: "Low",    label: "Low",    rule: "gt0" },
      { name: "Close",  label: "Close",  rule: "gt0" },
      { name: "Volume", label: "Volume", rule: "ge0" },
    ],
  },
  {
    title: "Daily movement",
    desc: "How much the price moved during the day.",
    fields: [
      { name: "Daily_Return",    label: "Daily return (%)", rule: "any" },
      { name: "Open_Close_Diff", label: "Open–close difference", rule: "any" },
      { name: "High_Low_Diff",   label: "High–low difference", rule: "any" },
    ],
  },
  {
    title: "Moving averages",
    desc: "Average closing price over recent days.",
    fields: [
      { name: "MA_10", label: "10-day average", rule: "gt0" },
      { name: "MA_20", label: "20-day average", rule: "gt0" },
      { name: "MA_50", label: "50-day average", rule: "gt0" },
    ],
  },
  {
    title: "Momentum",
    desc: "RSI and MACD show whether the trend is gaining or losing strength.",
    fields: [
      { name: "RSI",         label: "RSI",         rule: "pct" },
      { name: "MACD",        label: "MACD",        rule: "any" },
      { name: "MACD_Signal", label: "MACD signal", rule: "any" },
    ],
  },
  {
    title: "Bollinger bands",
    desc: "The price range around the 20-day average.",
    fields: [
      { name: "BB_Upper",  label: "Upper band",  rule: "any" },
      { name: "BB_Middle", label: "Middle band", rule: "any" },
      { name: "BB_Lower",  label: "Lower band",  rule: "any" },
    ],
  },
  {
    title: "Volume signals",
    fields: [
      { name: "Volume_Change", label: "Volume change (%)", rule: "any" },
      { name: "Volume_MA",     label: "Average volume",    rule: "ge0" },
    ],
  },
  {
    title: "Previous closes",
    desc: "Closing price a given number of trading days ago.",
    fields: [
      { name: "Close_Lag_1",  label: "1 day ago",  rule: "gt0" },
      { name: "Close_Lag_5",  label: "5 days ago", rule: "gt0" },
      { name: "Close_Lag_10", label: "10 days ago", rule: "gt0" },
      { name: "Close_Lag_20", label: "20 days ago", rule: "gt0" },
    ],
  },
  {
    title: "Market and economy",
    desc: "NIFTY index data and macro indicators.",
    fields: [
      { name: "NIFTY_Close",      label: "NIFTY close",       rule: "gt0" },
      { name: "NIFTY_Return",     label: "NIFTY return (%)",  rule: "any" },
      { name: "NIFTY_Volatility", label: "NIFTY volatility",  rule: "ge0" },
      { name: "CPI_Inflation",    label: "CPI inflation (%)", rule: "any" },
      { name: "GDP_Growth",       label: "GDP growth (%)",    rule: "any" },
    ],
  },
];

const RULES = {
  gt0: { hint: "Must be above 0" },
  ge0: { hint: "0 or higher" },
  pct: { hint: "Between 0 and 100" },
  any: { hint: "Any number" },
};

const ALL_FIELDS = FIELD_GROUPS.flatMap((g) => g.fields);

/* Sample row: the latest record in TCS_Final_ML_Dataset.csv */
const SAMPLE_DATA = {
  Open: 2065, High: 2085.1, Low: 2032.4, Close: 2032.4, Volume: 4437783,
  Daily_Return: -1.8496, Open_Close_Diff: -32.6, High_Low_Diff: 52.7,
  MA_10: 2107.92, MA_20: 2190.49, MA_50: 2282.51,
  RSI: 20.12, MACD: -64.55, MACD_Signal: -50.22,
  BB_Upper: 2391.33, BB_Middle: 2190.49, BB_Lower: 1989.64,
  Volume_Change: 80.61, Volume_MA: 3057435.4,
  Close_Lag_1: 2070.7, Close_Lag_5: 2105, Close_Lag_10: 2251, Close_Lag_20: 2399.3,
  NIFTY_Close: 22716.2, NIFTY_Return: -0.2812, NIFTY_Volatility: 0.6393,
  CPI_Inflation: 1.33, GDP_Growth: 7.41,
};

const VERDICT_TEXT = {
  BUY:  "The model reads these inputs as a buying signal.",
  HOLD: "The model reads these inputs as neutral: no strong move either way.",
  SELL: "The model reads these inputs as a selling signal.",
};

/* ----------------------------------------------------------
   2. DOM references
---------------------------------------------------------- */
const $ = (id) => document.getElementById(id);

const form        = $("predictForm");
const formFields  = $("formFields");
const formAlert   = $("formAlert");
const submitBtn   = $("submitBtn");
const submitLabel = $("submitLabel");
const sampleBtn   = $("sampleBtn");
const resetBtn    = $("resetBtn");
const retryBtn    = $("retryBtn");
const resultCard  = $("resultCard");
const apiStatus   = $("apiStatus");
const apiStatusText = $("apiStatusText");

/* ----------------------------------------------------------
   3. Build the form
---------------------------------------------------------- */
function buildForm() {
  const frag = document.createDocumentFragment();

  FIELD_GROUPS.forEach((group) => {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "group";

    const legend = document.createElement("legend");
    legend.textContent = group.title;
    fieldset.appendChild(legend);

    if (group.desc) {
      const desc = document.createElement("p");
      desc.className = "group__desc";
      desc.textContent = group.desc;
      fieldset.appendChild(desc);
    }

    const grid = document.createElement("div");
    grid.className = "group__grid";

    group.fields.forEach((f) => {
      const wrap = document.createElement("div");
      wrap.className = "field";
      wrap.dataset.field = f.name;

      const label = document.createElement("label");
      label.htmlFor = f.name;
      label.textContent = f.label;

      const input = document.createElement("input");
      input.type = "number";
      input.id = f.name;
      input.name = f.name;
      input.step = "any";
      input.inputMode = "decimal";
      input.autocomplete = "off";
      input.required = true;
      input.setAttribute("aria-describedby", `${f.name}-msg`);

      const msg = document.createElement("small");
      msg.id = `${f.name}-msg`;
      msg.className = "field__hint";
      msg.textContent = RULES[f.rule].hint;

      const err = document.createElement("small");
      err.className = "field__error";
      err.dataset.error = f.name;

      wrap.append(label, input, msg, err);
      grid.appendChild(wrap);

      input.addEventListener("input", () => clearFieldError(f.name));
      input.addEventListener("blur", () => {
        if (input.value !== "") showFieldError(f.name, validateField(f, input.value));
      });
    });

    fieldset.appendChild(grid);
    frag.appendChild(fieldset);
  });

  formFields.appendChild(frag);
}

/* ----------------------------------------------------------
   4. Validation (same limits as the Pydantic model)
---------------------------------------------------------- */
function validateField(field, raw) {
  if (raw === "" || raw === null) return "This field is required.";

  const value = Number(raw);
  if (!Number.isFinite(value)) return "Enter a valid number.";

  switch (field.rule) {
    case "gt0": if (!(value > 0))  return "Must be greater than 0."; break;
    case "ge0": if (!(value >= 0)) return "Must be 0 or higher.";    break;
    case "pct": if (value < 0 || value > 100) return "Must be between 0 and 100."; break;
  }
  return "";
}

function showFieldError(name, message) {
  const wrap = form.querySelector(`.field[data-field="${name}"]`);
  if (!wrap) return;
  wrap.classList.toggle("is-invalid", Boolean(message));
  wrap.querySelector(".field__error").textContent = message || "";
  wrap.querySelector("input").setAttribute("aria-invalid", message ? "true" : "false");
}

function clearFieldError(name) {
  showFieldError(name, "");
}

function clearAllErrors() {
  ALL_FIELDS.forEach((f) => clearFieldError(f.name));
  hideAlert();
}

/* Returns { errors: {name: msg}, payload: {name: number} } */
function validateForm() {
  const errors = {};
  const payload = {};

  ALL_FIELDS.forEach((f) => {
    const raw = form.elements[f.name].value.trim();
    const message = validateField(f, raw);
    if (message) errors[f.name] = message;
    else payload[f.name] = Number(raw);
  });

  return { errors, payload };
}

/* ----------------------------------------------------------
   5. Alerts and result states
---------------------------------------------------------- */
function showAlert(title, items = []) {
  formAlert.replaceChildren();

  const strong = document.createElement("strong");
  strong.textContent = title;
  formAlert.appendChild(strong);

  if (items.length) {
    const ul = document.createElement("ul");
    items.forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      ul.appendChild(li);
    });
    formAlert.appendChild(ul);
  }
  formAlert.hidden = false;
}

function hideAlert() {
  formAlert.hidden = true;
  formAlert.replaceChildren();
}

function setResultState(state, cls = "") {
  resultCard.dataset.state = state;
  if (cls) resultCard.dataset.class = cls;
  else delete resultCard.dataset.class;
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle("is-loading", isLoading);
  submitBtn.setAttribute("aria-busy", String(isLoading));
  submitLabel.textContent = isLoading ? "Predicting…" : "Get prediction";
  sampleBtn.disabled = isLoading;
  resetBtn.disabled = isLoading;
}

function showError(title, text) {
  $("errorTitle").textContent = title;
  $("errorText").textContent = text;
  setResultState("error");
}

function showResult(data) {
  const label = String(data.prediction || "").toUpperCase();

  $("verdict").textContent = label || "—";
  $("verdictText").textContent = VERDICT_TEXT[label] || "";
  $("metaModel").textContent = data.model ?? "—";
  $("metaFeatures").textContent = data.features_used ?? "—";
  $("metaNote").textContent = data.note ?? "";

  renderProbabilities(data.probabilities);
  setResultState("success", label);

  // Animate the bars after the card is visible
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll(".prob__fill").forEach((bar) => {
        bar.style.width = bar.dataset.width;
      });
    });
  });
}

function renderProbabilities(probabilities) {
  const wrap = $("probsWrap");
  const list = $("probsList");
  list.replaceChildren();

  if (!probabilities || typeof probabilities !== "object") {
    wrap.hidden = true;
    return;
  }

  Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, value]) => {
      const pct = Math.max(0, Math.min(100, value * 100));

      const li = document.createElement("li");

      const row = document.createElement("div");
      row.className = "prob__row";
      const n = document.createElement("span");
      n.textContent = name;
      const v = document.createElement("span");
      v.textContent = `${pct.toFixed(1)}%`;
      row.append(n, v);

      const track = document.createElement("div");
      track.className = "prob__track";
      const fill = document.createElement("div");
      fill.className = "prob__fill";
      fill.dataset.class = name.toUpperCase();
      fill.dataset.width = `${pct}%`;
      track.appendChild(fill);

      li.append(row, track);
      list.appendChild(li);
    });

  wrap.hidden = false;
}

/* ----------------------------------------------------------
   6. API calls
---------------------------------------------------------- */
async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/* Turn any FastAPI error body into { message, fieldErrors, list } */
async function parseApiError(response) {
  let body = null;
  try { body = await response.json(); } catch (_) { /* not JSON */ }

  const detail = body && body.detail;
  const result = { message: "", fieldErrors: {}, list: [] };

  // Pydantic validation errors: detail is an array
  if (Array.isArray(detail)) {
    detail.forEach((item) => {
      const loc = Array.isArray(item.loc) ? item.loc : [];
      const field = loc[loc.length - 1];
      const msg = item.msg || "Invalid value.";
      if (typeof field === "string" && ALL_FIELDS.some((f) => f.name === field)) {
        result.fieldErrors[field] = msg;
      }
      result.list.push(`${loc.slice(1).join(".") || "input"}: ${msg}`);
    });
    result.message = "The server rejected some of the values.";
    return result;
  }

  // Custom error from main.py: { message, features: [...] }
  if (detail && typeof detail === "object") {
    result.message = detail.message || "The request was rejected.";
    if (Array.isArray(detail.features)) result.list = detail.features;
    return result;
  }

  if (typeof detail === "string") {
    result.message = detail;
    return result;
  }

  result.message = `The server returned an unexpected response (HTTP ${response.status}).`;
  return result;
}

async function checkHealth() {
  apiStatus.dataset.state = "checking";
  apiStatusText.textContent = "Checking API…";

  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    apiStatus.dataset.state = "online";
    apiStatusText.textContent = `API online · ${data.feature_count} features`;
    return true;
  } catch (_) {
    apiStatus.dataset.state = "offline";
    apiStatusText.textContent = "API offline";
    return false;
  }
}

async function requestPrediction(payload) {
  const res = await fetchWithTimeout(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await parseApiError(res);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/* ----------------------------------------------------------
   7. Event handlers
---------------------------------------------------------- */
async function handleSubmit(event) {
  event.preventDefault();
  clearAllErrors();

  const { errors, payload } = validateForm();
  const names = Object.keys(errors);

  if (names.length) {
    names.forEach((name) => showFieldError(name, errors[name]));
    showAlert(
      `Fix ${names.length} ${names.length === 1 ? "field" : "fields"} before predicting.`
    );
    form.elements[names[0]].focus();
    return;
  }

  setLoading(true);
  setResultState("loading");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

  try {
    const data = await requestPrediction(payload);
    showResult(data);
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    apiStatus.dataset.state = "online";
  } catch (err) {
    handleRequestError(err);
  } finally {
    setLoading(false);
  }
}

function handleRequestError(err) {
  // Timeout
  if (err && err.name === "AbortError") {
    showError(
      "The request timed out",
      `The API did not respond within ${REQUEST_TIMEOUT_MS / 1000} seconds. Check that the server is running and try again.`
    );
    return;
  }

  // Network failure (server down, wrong port, CORS)
  if (err instanceof TypeError) {
    apiStatus.dataset.state = "offline";
    apiStatusText.textContent = "API offline";
    showError(
      "Can’t reach the API",
      `Make sure the backend is running at ${API_BASE_URL} (uvicorn main:app --port 8000), then try again.`
    );
    return;
  }

  // Error response from FastAPI
  if (err && err.status) {
    if (err.status === 422) {
      Object.entries(err.fieldErrors).forEach(([name, msg]) => showFieldError(name, msg));
      showAlert(err.message, err.list);
      showError("Some values were rejected", "Check the highlighted fields and submit again.");
      const first = Object.keys(err.fieldErrors)[0];
      if (first) form.elements[first].focus();
      return;
    }

    if (err.status >= 500) {
      showError(
        "The server hit an error",
        err.message || "Prediction failed. Check the backend logs for details."
      );
      return;
    }

    showError("Request failed", err.message || `HTTP ${err.status}`);
    return;
  }

  showError("Unexpected error", "Something went wrong. Please try again.");
  console.error(err);
}

function fillSample() {
  clearAllErrors();
  ALL_FIELDS.forEach((f) => {
    form.elements[f.name].value = SAMPLE_DATA[f.name];
  });
}

function resetAll() {
  form.reset();
  clearAllErrors();
  setResultState("idle");
}

/* ----------------------------------------------------------
   8. Init
---------------------------------------------------------- */
buildForm();
form.addEventListener("submit", handleSubmit);
sampleBtn.addEventListener("click", fillSample);
resetBtn.addEventListener("click", resetAll);
retryBtn.addEventListener("click", () => form.requestSubmit());
apiStatus.addEventListener("click", checkHealth);
checkHealth();
