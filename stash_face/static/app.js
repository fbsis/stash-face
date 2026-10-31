"use strict";

const $ = id => document.getElementById(id);
let token = "";
let currentFilter = "all";
let currentSearch = "";
let selectedPerson = null;
let allPeople = [];
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
  $("people-grid").replaceChildren();
  notify("Disconnected. Token forgotten.");
}

function updateStats(stats) {
  if (!stats) return;
  $("stat-people").textContent = stats.clusters || 0;
  $("stat-faces").textContent = stats.faces || 0;
  $("stat-images").textContent = stats.images || 0;
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
    await api("/jobs", "POST", { action: "cluster" });
    notify("DBSCAN re-clustering queued! Refreshing in a few moments...");
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
