"use strict";

const $ = id => document.getElementById(id);
let token = "";
let currentFilter = "all";
let currentSearch = "";
let selectedPerson = null;
let allPeople = [];
let progressTimer = null;
let selectedJobId = null;
let facesOffset = 0;
const cropUrls = new Set();

async function setCropImage(img, filename) {
  if (!filename) return;
  try {
    const response = await fetch(`/crops/${encodeURIComponent(filename)}`, {
      headers: { "X-API-Token": token }, cache: "no-store"
    });
    if (!response.ok) return;
    const url = URL.createObjectURL(await response.blob());
    cropUrls.add(url);
    img.src = url;
  } catch (_) {
    // The card remains usable if an individual crop is unavailable.
  }
}

const notify = (text, timeout = 4000) => {
  const banner = $("status-banner");
  const msg = $("status-message");
  msg.textContent = text;
  banner.hidden = false;
  if (timeout > 0) {
    setTimeout(() => { banner.hidden = true; }, timeout);
  }
};

async function api(path, method = "GET", data) {
  if (!token) throw new Error("Not connected. Enter service token first.");
  const response = await fetch(path, {
    method,
    headers: {
      "X-API-Token": token,
      "Content-Type": "application/json"
    },
    body: data === undefined ? undefined : JSON.stringify(data),
    cache: "no-store"
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  }
  return body;
}

async function connect(provided) {
  if (!provided) return;
  token = provided.trim();
  $("token").value = token;

  try {
    const health = await api("/health");
    localStorage.setItem("stash_face_token", token);

    $("connect-form").hidden = true;
    $("action-controls").hidden = false;
    $("stats-bar").hidden = false;
    $("workspace").hidden = false;

    updateStats(health.stats);
    notify(`Connected to Stash Face (${health.detector.model_name})`);

    await loadPeople();
    await loadProgress();
    await loadJobs();
    clearInterval(progressTimer);
    progressTimer = setInterval(() => { loadProgress(); loadJobs(); if (!$("media-panel").hidden) loadMedia(); if (!$("suggestions-panel").hidden) loadSuggestions(); if (!$("statistics-panel").hidden) loadStatistics(); }, 3000);
  } catch (err) {
    localStorage.removeItem("stash_face_token");
    notify("Connection failed: " + err.message);
  }
}

function disconnect() {
  localStorage.removeItem("stash_face_token");
  token = "";
  $("token").value = "";
  $("connect-form").hidden = false;
  $("action-controls").hidden = true;
  $("stats-bar").hidden = true;
  $("workspace").hidden = true;
  clearInterval(progressTimer);
  progressTimer = null;
  $("people-grid").replaceChildren();
  notify("Disconnected. Token forgotten.");
}

function updateStats(stats) {
  if (!stats) return;
  $("stat-people").textContent = stats.clusters || 0;
  $("stat-faces").textContent = stats.faces || 0;
  $("stat-images").textContent = stats.images || 0;
  $("stat-progress").textContent = `${stats.media_percent || 0}%`;
}

async function loadProgress() {
  try {
    const progress = await api("/progress");
    $("stat-progress").textContent = `${progress.percent || 0}%`;
    const queued = Object.entries(progress.jobs || {}).filter(([key]) => key.endsWith("_queued") || key.endsWith("_retry"))
      .reduce((sum, [, count]) => sum + count, 0);
    const running = Object.entries(progress.jobs || {}).filter(([key]) => key.endsWith("_running"))
      .reduce((sum, [, count]) => sum + count, 0);
    $("media-progress").textContent = `${progress.media_completed || 0} / ${progress.media_total || 0} media indexed (${progress.percent || 0}%). ` +
      `${queued} queued, ${running} running, ${progress.statuses?.unindexed || 0} unindexed, ${progress.statuses?.failed || 0} failed.`;
    const clustering = progress.clustering || {};
    const clusterResult = clustering.result || {};
    const clusterFaces = clusterResult.faces_evaluated;
    const inferredState = clustering.state ||
      (progress.jobs?.cluster_running ? "running" :
        progress.jobs?.cluster_queued || progress.jobs?.cluster_retry ? "queued" : "unknown");
    const clusterText = inferredState === "running"
      ? `DBSCAN: running${clustering.detail ? ` · ${clustering.detail}` : ""}`
      : inferredState === "queued" || inferredState === "retry"
        ? `DBSCAN: queued (${inferredState})`
        : inferredState === "done"
          ? `DBSCAN: completed${clusterFaces !== undefined ? ` · ${clusterFaces} faces evaluated` : ""}`
          : inferredState === "failed"
            ? `DBSCAN: failed · ${clustering.error || "see job details"}`
            : "DBSCAN: status unavailable";
    $("cluster-progress").textContent = clusterText;
  } catch (err) {
    notify("Error loading progress: " + err.message);
  }
}

function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

function renderStatisticsList(id, rows) {
  const list = $(id);
  list.replaceChildren();
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    list.append(term, description);
  }
}

function renderStatisticsCards(cards) {
  const container = $("statistics-cards");
  container.replaceChildren();
  for (const [label, value] of cards) {
    const card = document.createElement("div");
    card.className = "statistics-card";
    const number = document.createElement("strong");
    number.textContent = value;
    const caption = document.createElement("span");
    caption.textContent = label;
    card.append(number, caption);
    container.append(card);
  }
}

async function loadStatistics() {
  try {
    const stats = await api("/statistics");
    const media = stats.media || {};
    const faces = stats.faces || {};
    const clusters = stats.clusters || {};
    const performance = stats.performance || {};
    const durations = performance.duration_seconds || {};
    const allDuration = durations.all || {};
    const imageDuration = durations.image || {};
    const videoDuration = durations.video || {};
    const clustering = stats.clustering || {};
    const result = clustering.result || {};
    const imageMedia = media.by_type?.image || {};
    const videoMedia = media.by_type?.video || {};
    const imageCompleted = (imageMedia.indexed || 0) + (imageMedia.clustered || 0) + (imageMedia.approved || 0) + (imageMedia.no_faces || 0) + (imageMedia.ignored || 0);
    const videoCompleted = (videoMedia.indexed || 0) + (videoMedia.clustered || 0) + (videoMedia.approved || 0) + (videoMedia.no_faces || 0) + (videoMedia.ignored || 0);

    renderStatisticsCards([
      ["Active faces", faces.active || 0],
      ["Faces without a cluster", faces.unclustered || 0],
      ["Clusters", clusters.total || 0],
      ["Media indexed", `${media.percent || 0}%`],
      ["Median job duration", formatSeconds(allDuration.median_seconds)],
      ["Jobs completed in the last hour", Object.values(performance.completed_last_hour || {}).reduce((sum, count) => sum + count, 0)],
    ]);
    renderStatisticsList("statistics-media", [
      ["Total media", media.total || 0],
      ["Completed media", `${media.completed || 0} (${media.percent || 0}%)`],
      ["Photos", `${imageCompleted} completed / ${imageMedia.unindexed || 0} unindexed`],
      ["Videos", `${videoCompleted} completed / ${videoMedia.unindexed || 0} unindexed`],
      ["Media with faces", faces.media_with_faces || 0],
      ["Average faces per media", faces.average_per_media ?? "—"],
      ["Median faces per media", faces.median_per_media ?? "—"],
    ]);
    renderStatisticsList("statistics-clusters", [
      ["Active faces", faces.active || 0],
      ["Faces in clusters", faces.clustered || 0],
      ["Faces without a cluster", faces.unclustered || 0],
      ["Named clusters", clusters.named || 0],
      ["Unnamed clusters", clusters.unnamed || 0],
      ["Largest cluster", clusters.largest_size || 0],
      ["Median cluster size", clusters.median_size ?? "—"],
    ]);
    renderStatisticsList("statistics-performance", [
      ["Photo job average", formatSeconds(imageDuration.average_seconds)],
      ["Photo job median", formatSeconds(imageDuration.median_seconds)],
      ["Video job average", formatSeconds(videoDuration.average_seconds)],
      ["Video job median", formatSeconds(videoDuration.median_seconds)],
      ["All jobs p95", formatSeconds(allDuration.p95_seconds)],
      ["Photo throughput (last hour)", `${performance.throughput_per_minute_last_hour?.image || 0}/min`],
      ["Video throughput (last hour)", `${performance.throughput_per_minute_last_hour?.video || 0}/min`],
    ]);
    renderStatisticsList("statistics-clustering", [
      ["State", clustering.state || "unknown"],
      ["Stage", clustering.stage || "—"],
      ["Detail", clustering.detail || "—"],
      ["Faces evaluated", result.faces_evaluated ?? "—"],
      ["Clusters created", result.clusters_created ?? "—"],
      ["Performers created", result.performers_created ?? "—"],
      ["Suggestions created", result.suggestions_created ?? "—"],
    ]);
    const generated = stats.generated_at ? new Date(stats.generated_at * 1000).toLocaleString() : "unknown";
    $("statistics-updated").textContent = `Updated ${generated}. Job durations include queue wait and execution time.`;
  } catch (err) {
    $("statistics-updated").textContent = `Statistics unavailable: ${err.message}`;
  }
}

function jobLabel(job) {
  const payload = job.payload || {};
  if (job.kind === "image") return `Photo ${payload.image_id || "unknown"}`;
  if (job.kind === "video") return `Video ${payload.scene_id || "unknown"}`;
  if (job.kind === "scan") return "Library scan";
  if (job.kind === "cluster") return "DBSCAN clustering";
  if (job.kind === "sync_performers") return "Performer reference sync";
  return `${job.kind} job`;
}

function jobSteps(kind) {
  return {
    scan: ["enumerate media", "queue files"],
    image: ["download photo", "detect and embed faces", "save crop and embedding", "tag Performer"],
    video: ["extract three frames", "detect and embed faces on GPU", "save frame results", "tag scene"],
    cluster: ["load embeddings", "DBSCAN", "preserve links"],
    sync_performers: ["download references", "detect face", "save reference"],
    auto_tag: ["read file paths", "match existing names", "apply Stash metadata"],
  }[kind] || ["process"];
}

function showJobDetails(job) {
  selectedJobId = job.id;
  $("job-detail").hidden = false;
  $("job-detail-content").textContent = JSON.stringify({
    id: job.id, kind: job.kind, state: job.state, stage: job.stage,
    detail: job.detail, attempts: job.attempts,
    created_at: new Date(job.created_at * 1000).toISOString(),
    updated_at: new Date(job.updated_at * 1000).toISOString(),
    payload: job.payload, result: job.result, error: job.error,
  }, null, 2);
}

async function loadJobs() {
  try {
    const jobs = await api("/jobs?limit=200");
    const tree = $("job-tree");
    const selected = selectedJobId;
    tree.replaceChildren();
    if (!jobs.length) { tree.textContent = "No jobs have been queued."; return; }
    const groups = new Map();
    for (const job of jobs) {
      if (!groups.has(job.kind)) groups.set(job.kind, []);
      groups.get(job.kind).push(job);
    }
    for (const [kind, items] of groups) {
      const group = document.createElement("section"); group.className = "pipeline-group";
      const title = document.createElement("div"); title.className = "pipeline-title";
      const active = items.filter(item => ["queued", "running", "retry"].includes(item.state)).length;
      title.textContent = `${kind} pipeline`;
      const count = document.createElement("span"); count.textContent = `${active} active / ${items.length} shown`;
      title.append(count); group.append(title);
      for (const job of items) {
        const node = document.createElement("button");
        node.type = "button"; node.className = `pipeline-node ${job.state} ${job.state === "running" ? "active" : ""}`;
        if (job.id === selected) node.classList.add("selected");
        const dot = document.createElement("span"); dot.className = "node-dot";
        const main = document.createElement("span"); main.className = "node-main";
        const label = document.createElement("span"); label.className = "node-label"; label.textContent = jobLabel(job);
        const sub = document.createElement("span"); sub.className = "node-subtitle"; sub.textContent = `${job.stage || job.state}${job.detail ? ` · ${job.detail}` : ""}`;
        main.append(label, sub);
        const state = document.createElement("span"); state.className = "node-state"; state.textContent = job.state;
        node.append(dot, main, state); node.onclick = () => showJobDetails(job); group.append(node);
      }
      tree.append(group);
    }
    $("job-refresh-state").textContent = `Live · ${jobs.length} jobs`;
    if (selected) {
      const current = jobs.find(job => job.id === selected);
      if (current) showJobDetails(current);
    }
  } catch (err) {
    $("job-refresh-state").textContent = "Unavailable";
  }
}

async function loadFaces(reset = true) {
  try {
    if (reset) { facesOffset = 0; $("faces-list").replaceChildren(); }
    const filter = $("faces-filter").value;
    const q = $("faces-search").value.trim();
    const result = await api(`/faces?filter=${encodeURIComponent(filter)}&q=${encodeURIComponent(q)}&limit=100&offset=${facesOffset}`);
    const list = $("faces-list");
    if (reset && !result.items.length) list.textContent = "No faces match this filter.";
    for (const face of result.items) {
      const card = document.createElement("article"); card.className = "face-card";
      const img = document.createElement("img"); img.loading = "lazy"; img.alt = "Detected face";
      setCropImage(img, face.crop);
      const info = document.createElement("div"); info.className = "face-card-info";
      const title = document.createElement("div"); title.className = "face-card-title";
      title.textContent = face.cluster_name || "Not grouped";
      const state = document.createElement("div"); state.className = "face-card-state";
      state.textContent = face.performer_id || face.cluster_performer_id ? `Linked Performer #${face.performer_id || face.cluster_performer_id}` : "Unknown identity";
      const source = document.createElement("div"); source.className = "face-card-source";
      source.textContent = `${face.media_type || "image"} · ${face.title || face.source_id || `ID ${face.image_id}`} · score ${Number(face.score || 0).toFixed(2)}`;
      info.append(title, state, source); card.append(img, info); list.append(card);
    }
    facesOffset += result.items.length;
    $("faces-progress").textContent = `${facesOffset} of ${result.total} faces shown`;
    $("faces-load-more").hidden = facesOffset >= result.total;
  } catch (err) {
    notify("Error loading faces: " + err.message);
  }
}

async function loadMedia() {
  try {
    const status = $("media-status-filter").value;
    const mediaType = $("media-type-filter").value;
    const result = await api(`/media?status=${encodeURIComponent(status)}&media_type=${encodeURIComponent(mediaType)}&limit=100`);
    const list = $("media-list");
    list.replaceChildren();
    if (!result.items.length) {
      list.textContent = "No media matches this filter.";
      return;
    }
    for (const item of result.items) {
      const row = document.createElement("div");
      row.className = "media-row";
      const kind = document.createElement("span"); kind.className = "media-kind"; kind.textContent = item.media_type;
      const state = document.createElement("span"); state.className = "media-state"; state.textContent = item.status;
      const title = document.createElement("span"); title.className = "media-title"; title.textContent = item.title || item.source_id;
      const source = document.createElement("span"); source.className = "media-source"; source.textContent = item.source_id;
      const frame = document.createElement("span"); frame.textContent = item.media_type === "video" ? `frame ${Number(item.frame_index || 0) + 1}/3` : "photo";
      row.append(kind, state, title, source, frame); list.append(row);
    }
  } catch (err) {
    notify("Error loading media index: " + err.message);
  }
}

async function loadSuggestions() {
  try {
    const status = $("suggestions-status").value;
    const result = await api(`/suggestions?status=${encodeURIComponent(status)}&limit=100`);
    const list = $("suggestions-list");
    list.replaceChildren();
    if (!result.items.length) {
      list.textContent = "No suggestions to review.";
      $("suggestions-progress").textContent = "0 suggestions";
      return;
    }
    for (const item of result.items) {
      const card = document.createElement("article");
      card.className = "suggestion-card";
      if (item.exemplar_crop) {
        const img = document.createElement("img"); img.className = "suggestion-avatar";
        img.alt = "Suggested person"; setCropImage(img, item.exemplar_crop); card.append(img);
      }
      const body = document.createElement("div"); body.className = "suggestion-body";
      const title = document.createElement("strong");
      title.textContent = `${item.media_type === "video" ? "Video" : "Photo"}: ${item.title || item.source_id}`;
      const target = document.createElement("div");
      target.textContent = `Suggested person: ${item.cluster_name || `#${item.cluster_id}`} (${Math.round(item.score * 100)}%)`;
      const folder = document.createElement("div"); folder.className = "suggestion-reason";
      let reason = {};
      try { reason = JSON.parse(item.reason || "{}"); } catch (_) {}
      folder.textContent = `Same folder · ${reason.folder_identities || "?"} identities found`;
      body.append(title, target, folder);
      if (item.status === "pending") {
        const actions = document.createElement("div"); actions.className = "suggestion-actions";
        for (const [decision, label, cls] of [["accepted", "Accept", "btn-primary"], ["rejected", "Reject", "btn-ghost"]]) {
          const button = document.createElement("button"); button.className = `btn btn-small ${cls}`; button.textContent = label;
          button.onclick = async () => {
            try { await api(`/suggestions/${item.id}/decision`, "POST", { decision }); notify(`Suggestion ${label.toLowerCase()}ed.`); loadSuggestions(); }
            catch (err) { notify("Suggestion failed: " + err.message); }
          };
          actions.append(button);
        }
        body.append(actions);
      } else {
        const state = document.createElement("span"); state.className = "badge"; state.textContent = item.status; body.append(state);
      }
      card.append(body); list.append(card);
    }
    $("suggestions-progress").textContent = `${result.total} suggestions`;
  } catch (err) { notify("Error loading suggestions: " + err.message); }
}

async function loadPeople() {
  try {
    const query = new URLSearchParams({ q: currentSearch, filter: currentFilter });
    allPeople = await api("/people?" + query.toString());
    renderPeople();
  } catch (err) {
    notify("Error loading people: " + err.message);
  }
}

function renderPeople() {
  const grid = $("people-grid");
  const empty = $("empty-state");
  for (const url of cropUrls) URL.revokeObjectURL(url);
  cropUrls.clear();
  grid.replaceChildren();

  if (!allPeople.length) {
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  for (const person of allPeople) {
    const card = document.createElement("div");
    card.className = "person-card";

    // Avatar
    const avatarWrap = document.createElement("div");
    avatarWrap.className = "person-avatar-wrap";
    const avatar = document.createElement("img");
    avatar.className = "person-avatar";
    avatar.loading = "lazy";
    avatar.alt = person.name || "Person face";
    if (person.exemplar_crop) setCropImage(avatar, person.exemplar_crop);
    avatarWrap.appendChild(avatar);

    // Name
    const nameEl = document.createElement("div");
    nameEl.className = "person-name";
    nameEl.textContent = person.name || `Person #${person.id}`;
    nameEl.title = person.name;

    // Photos count
    const countEl = document.createElement("div");
    countEl.className = "person-count";
    countEl.textContent = `${person.face_count || 0} photos`;

    card.append(avatarWrap, nameEl, countEl);

    // Stash linked badge
    if (person.performer_id) {
      const badge = document.createElement("span");
      badge.className = "person-badge";
      badge.textContent = "Stash ✓";
      card.appendChild(badge);
    }

    card.onclick = () => openPersonModal(person.id);
    grid.appendChild(card);
  }
}

async function openPersonModal(clusterId) {
  try {
    selectedPerson = await api(`/people/${clusterId}`);
    $("modal-avatar").src = "";
    if (selectedPerson.exemplar_crop) setCropImage($("modal-avatar"), selectedPerson.exemplar_crop);
    $("modal-name-input").value = selectedPerson.name || "";
    $("modal-face-count").textContent = `${selectedPerson.face_count || 0} photos (${selectedPerson.image_count || 0} images)`;

    if (selectedPerson.performer_id) {
      $("modal-performer-badge").hidden = false;
      $("modal-performer-badge").textContent = `Stash Performer #${selectedPerson.performer_id}`;
    } else {
      $("modal-performer-badge").hidden = true;
    }

    // Populate merge select with other people
    const mergeSelect = $("modal-merge-select");
    mergeSelect.replaceChildren();
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = "Merge into...";
    mergeSelect.appendChild(defaultOpt);

    for (const p of allPeople) {
      if (p.id !== selectedPerson.id) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.name || `Person #${p.id}`;
        mergeSelect.appendChild(opt);
      }
    }

    // Render faces grid
    const facesGrid = $("modal-faces-grid");
    facesGrid.replaceChildren();
    for (const face of (selectedPerson.faces || [])) {
      const thumb = document.createElement("div");
      thumb.className = "face-thumb";
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "Detected face";
      setCropImage(img, face.crop);
      img.title = `Image ID: ${face.image_id}`;
      thumb.appendChild(img);
      facesGrid.appendChild(thumb);
    }

    $("person-modal").hidden = false;
  } catch (err) {
    notify("Error opening person details: " + err.message);
  }
}

function closePersonModal() {
  $("person-modal").hidden = true;
  selectedPerson = null;
}

// Event Listeners
$("connect-form").onsubmit = event => {
  event.preventDefault();
  connect($("token").value);
};

$("btn-disconnect").onclick = disconnect;

// Action Buttons
$("btn-scan").onclick = async () => {
  try {
    await api("/jobs", "POST", { action: "scan" });
    notify("Library scan started! Images are being processed in background.");
  } catch (err) {
    notify("Failed to start scan: " + err.message);
  }
};

$("btn-cluster").onclick = async () => {
  try {
    const result = await api("/jobs", "POST", { action: "cluster", force: true });
    notify(result.created
      ? "Final DBSCAN clustering forced for the currently indexed media."
      : "A DBSCAN clustering job is already running or queued.");
    setTimeout(loadPeople, 3000);
  } catch (err) {
    notify("Failed to start clustering: " + err.message);
  }
};

$("btn-sync-performers").onclick = async () => {
  try {
    await api("/jobs", "POST", { action: "sync_performers" });
    notify("Syncing Performers and face embeddings from Stash...");
  } catch (err) {
    notify("Failed to sync performers: " + err.message);
  }
};

// People / Media / All Faces views
function setView(view) {
  const people = view === "people", media = view === "media", suggestions = view === "suggestions", statistics = view === "statistics";
  $("view-people").classList.toggle("active", people);
  $("view-media").classList.toggle("active", media);
  $("view-faces").classList.toggle("active", view === "faces");
  $("view-suggestions").classList.toggle("active", suggestions);
  $("view-statistics").classList.toggle("active", statistics);
  $("people-section").hidden = !people;
  $("media-panel").hidden = !media;
  $("faces-panel").hidden = view !== "faces";
  $("suggestions-panel").hidden = !suggestions;
  $("statistics-panel").hidden = !statistics;
}
$("view-people").onclick = () => setView("people");
$("view-media").onclick = () => { setView("media"); loadMedia(); loadProgress(); };
$("view-faces").onclick = () => { setView("faces"); loadFaces(); };
$("view-suggestions").onclick = () => { setView("suggestions"); loadSuggestions(); };
$("view-statistics").onclick = () => { setView("statistics"); loadStatistics(); };
$("stats-refresh").onclick = loadStatistics;
$("suggestions-status").onchange = loadSuggestions;
$("media-status-filter").onchange = loadMedia;
$("media-type-filter").onchange = loadMedia;
$("faces-filter").onchange = () => loadFaces(true);
$("faces-search").oninput = () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => loadFaces(true), 300);
};
$("faces-load-more").onclick = () => loadFaces(false);
$("close-job-detail").onclick = () => { $("job-detail").hidden = true; selectedJobId = null; };

// Filter Pills
document.querySelectorAll(".filter-pills .pill").forEach(pill => {
  pill.onclick = () => {
    document.querySelectorAll(".filter-pills .pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    currentFilter = pill.dataset.filter;
    loadPeople();
  };
});

let searchDebounce = null;
$("search-input").oninput = event => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => {
    currentSearch = event.target.value.trim();
    loadPeople();
  }, 300);
};

// Modal Actions
$("modal-close").onclick = closePersonModal;
$("modal-backdrop").onclick = closePersonModal;

$("modal-save-name-btn").onclick = async () => {
  if (!selectedPerson) return;
  const newName = $("modal-name-input").value.trim();
  if (!newName) return;
  try {
    await api(`/people/${selectedPerson.id}`, "PATCH", { name: newName });
    notify("Name updated successfully!");
    await loadPeople();
  } catch (err) {
    notify("Error saving name: " + err.message);
  }
};

$("modal-create-performer-btn").onclick = async () => {
  if (!selectedPerson) return;
  const name = $("modal-name-input").value.trim();
  if (!name) {
    notify("Please enter a name for the person first.");
    return;
  }
  if (!confirm(`Create Performer '${name}' in Stash and assign to all photos?`)) return;

  try {
    const res = await api(`/people/${selectedPerson.id}/create-performer`, "POST", { name });
    notify(`Performer created in Stash (#${res.performer_id}) and assigned to ${res.tagged_images} images!`);
    await openPersonModal(selectedPerson.id);
    await loadPeople();
  } catch (err) {
    notify("Error creating performer in Stash: " + err.message);
  }
};

$("modal-link-performer-btn").onclick = async () => {
  if (!selectedPerson) return;
  const query = $("modal-performer-search").value.trim();
  if (!query) return;

  try {
    const res = await api(`/performers?q=${encodeURIComponent(query)}`);
    const performers = res.performers || [];
    if (!performers.length) {
      notify(`No Performer found matching '${query}' in Stash.`);
      return;
    }

    const matched = performers[0];
    if (!confirm(`Link to Performer '${matched.name}' (#${matched.id}) and tag all photos in Stash?`)) return;

    const linkRes = await api(`/people/${selectedPerson.id}/link-performer`, "POST", {
      performer_id: String(matched.id),
      performer_name: matched.name
    });
    notify(`Linked successfully! ${linkRes.tagged_images} images tagged in Stash.`);
    await openPersonModal(selectedPerson.id);
    await loadPeople();
  } catch (err) {
    notify("Error linking performer: " + err.message);
  }
};

$("modal-merge-btn").onclick = async () => {
  if (!selectedPerson) return;
  const targetId = $("modal-merge-select").value;
  if (!targetId) {
    notify("Please select another person to merge into.");
    return;
  }
  if (!confirm("Are you sure you want to merge this person? This action cannot be undone.")) return;

  try {
    await api(`/people/${selectedPerson.id}/merge`, "POST", { target_id: Number(targetId) });
    notify("People merged successfully!");
    closePersonModal();
    await loadPeople();
  } catch (err) {
    notify("Error merging people: " + err.message);
  }
};

$("modal-hide-btn").onclick = async () => {
  if (!selectedPerson) return;
  try {
    const isHidden = selectedPerson.hidden === 1;
    await api(`/people/${selectedPerson.id}`, "PATCH", { hidden: !isHidden });
    notify(isHidden ? "Person unhidden!" : "Person hidden!");
    closePersonModal();
    await loadPeople();
  } catch (err) {
    notify("Error hiding person: " + err.message);
  }
};

// Auto-connect on page load if token is stored in localStorage
document.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("stash_face_token");
  if (saved) {
    connect(saved);
  }
});
