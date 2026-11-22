(function () {
  "use strict";

  // Exact pixel calibration measured from colorbar-free 699x307 STFT PNGs (plot box X=53..687, Y=13..248)
  const PLOT_X0 = 53 / 699;
  const PLOT_X1 = 687 / 699;
  const PLOT_SPAN_X = PLOT_X1 - PLOT_X0;

  const STREAM_META = {
    acs:      { short: "Raw ACS",       full: "Raw ACS (Air-Conduction)" },
    bcs:      { short: "Raw BCS",       full: "Raw BCS (Bone-Conduction)" },
    earphone: { short: "Sony XM4",      full: "Sony WF-1000XM4" },
    enhanced: { short: "BAFNet+ 50 ms", full: "BAFNet+ On-Device 50 ms" }
  };

  const TABLE_DATA = {
    s1: {
      metrics: {
        PESQ: {
          snrs: ["\u221220 dB", "\u221210 dB", "0 dB", "+10 dB", "+15 dB"],
          rows: [
            { model: "Attn-DC-CRN EF", vals: [1.422, 1.987, 2.508, 2.880, 3.004], bold: [0, 0, 0, 0, 0] },
            { model: "DBMIF",          vals: [1.260, 1.520, 1.959, 2.549, 2.863], bold: [0, 0, 0, 0, 0] },
            { model: "MMINet",         vals: [1.258, 1.619, 2.117, 2.661, 2.945], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet",         vals: [2.295, 2.590, 3.025, 3.521, 3.753], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet_B+",      vals: [2.827, 3.065, 3.388, 3.782, 3.975], bold: [1, 0, 0, 0, 0] },
            { model: "BAFNet+ (ours)", vals: [2.803, 3.111, 3.485, 3.875, 4.048], bold: [0, 1, 1, 1, 1], highlight: true }
          ]
        },
        STOI: {
          snrs: ["\u221220 dB", "\u221210 dB", "0 dB", "+10 dB", "+15 dB"],
          rows: [
            { model: "Attn-DC-CRN EF", vals: [0.783, 0.878, 0.926, 0.950, 0.956], bold: [0, 0, 0, 0, 0] },
            { model: "DBMIF",          vals: [0.757, 0.842, 0.915, 0.962, 0.976], bold: [0, 0, 0, 0, 0] },
            { model: "MMINet",         vals: [0.771, 0.870, 0.929, 0.965, 0.976], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet",         vals: [0.913, 0.930, 0.956, 0.979, 0.987], bold: [1, 0, 0, 0, 0] },
            { model: "BAFNet_B+",      vals: [0.913, 0.931, 0.955, 0.977, 0.986], bold: [1, 0, 0, 0, 0] },
            { model: "BAFNet+ (ours)", vals: [0.910, 0.933, 0.959, 0.981, 0.988], bold: [0, 1, 1, 1, 1], highlight: true }
          ]
        },
        CSIG: {
          snrs: ["\u221220 dB", "\u221210 dB", "0 dB", "+10 dB", "+15 dB"],
          rows: [
            { model: "Attn-DC-CRN EF", vals: [3.055, 3.715, 4.200, 4.553, 4.671], bold: [0, 0, 0, 0, 0] },
            { model: "DBMIF",          vals: [2.552, 2.969, 3.480, 4.058, 4.325], bold: [0, 0, 0, 0, 0] },
            { model: "MMINet",         vals: [2.610, 3.133, 3.703, 4.280, 4.545], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet",         vals: [3.655, 4.130, 4.546, 4.893, 4.977], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet_B+",      vals: [3.984, 4.361, 4.712, 4.953, 4.994], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet+ (ours)", vals: [4.161, 4.480, 4.792, 4.979, 4.999], bold: [1, 1, 1, 1, 1], highlight: true }
          ]
        },
        CBAK: {
          snrs: ["\u221220 dB", "\u221210 dB", "0 dB", "+10 dB", "+15 dB"],
          rows: [
            { model: "Attn-DC-CRN EF", vals: [2.355, 2.722, 3.038, 3.257, 3.329], bold: [0, 0, 0, 0, 0] },
            { model: "DBMIF",          vals: [2.255, 2.575, 3.016, 3.572, 3.868], bold: [0, 0, 0, 0, 0] },
            { model: "MMINet",         vals: [2.348, 2.652, 2.985, 3.352, 3.545], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet",         vals: [3.044, 3.326, 3.725, 4.230, 4.485], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet_B+",      vals: [3.119, 3.464, 3.827, 4.270, 4.497], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet+ (ours)", vals: [3.130, 3.520, 3.912, 4.348, 4.557], bold: [1, 1, 1, 1, 1], highlight: true }
          ]
        },
        COVL: {
          snrs: ["\u221220 dB", "\u221210 dB", "0 dB", "+10 dB", "+15 dB"],
          rows: [
            { model: "Attn-DC-CRN EF", vals: [2.270, 2.892, 3.397, 3.763, 3.885], bold: [0, 0, 0, 0, 0] },
            { model: "DBMIF",          vals: [1.944, 2.287, 2.763, 3.356, 3.651], bold: [0, 0, 0, 0, 0] },
            { model: "MMINet",         vals: [1.973, 2.423, 2.959, 3.527, 3.810], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet",         vals: [2.971, 3.406, 3.852, 4.332, 4.546], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet_B+",      vals: [3.460, 3.776, 4.142, 4.539, 4.717], bold: [0, 0, 0, 0, 0] },
            { model: "BAFNet+ (ours)", vals: [3.538, 3.863, 4.243, 4.622, 4.778], bold: [1, 1, 1, 1, 1], highlight: true }
          ]
        }
      }
    },
    s5: {
      tiers: [
        {
          title: "T_alg = 12.5 ms",
          rows: [
            { model: "Mapping branch", vals: [3.126, 3.126, 3.126, 3.126, 3.126], bold: [1, 1, 0, 0, 0] },
            { model: "Masking branch", vals: [1.780, 2.400, 3.036, 3.567, 3.784], bold: [0, 0, 0, 0, 0] },
            { model: "Full BAFNet+",   vals: [2.846, 3.085, 3.364, 3.703, 3.877], bold: [0, 0, 1, 1, 1], highlight: true }
          ]
        },
        {
          title: "T_alg = 50 ms",
          rows: [
            { model: "Mapping branch", vals: [3.141, 3.141, 3.141, 3.141, 3.141], bold: [1, 0, 0, 0, 0] },
            { model: "Masking branch", vals: [1.899, 2.572, 3.214, 3.713, 3.903], bold: [0, 0, 0, 0, 0] },
            { model: "Full BAFNet+",   vals: [2.938, 3.168, 3.441, 3.770, 3.933], bold: [0, 1, 1, 1, 1], highlight: true }
          ]
        },
        {
          title: "T_alg = 100 ms",
          rows: [
            { model: "Mapping branch", vals: [3.164, 3.164, 3.164, 3.164, 3.164], bold: [1, 0, 0, 0, 0] },
            { model: "Masking branch", vals: [1.887, 2.566, 3.211, 3.705, 3.898], bold: [0, 0, 0, 0, 0] },
            { model: "Full BAFNet+",   vals: [2.935, 3.168, 3.442, 3.773, 3.939], bold: [0, 1, 1, 1, 1], highlight: true }
          ]
        }
      ]
    }
  };

  const QUEUED_INFO = {
    taps: {
      title: "Section 1 · TAPS Paired Multi-SNR Benchmark (\u221220 / \u221210 / 0 / +10 dB)",
      desc: "Paired 7-system comparison (Noisy ACS, Raw BCS, Attn-DC-CRN EF, BiNet Two-Tower, BAFNet IS'25, BAFNet+ 50 ms DSP, Clean Reference) is scheduled for batch GPU rendering. Quantitative PESQ, STOI, CSIG, CBAK, and COVL scores across SNRs are available in Table S1 below.",
      tab: "s1"
    },
    drift: {
      title: "Section 2 · E1 Time-Varying Gain Drift Benchmark (\u00b110 dB Step, Ramp, Sinusoid)",
      desc: "Controlled BCS-to-ACS time-varying level drift at \u221210 dB and 0 dB input SNR comparing Full BAFNet+ against No-calibration and Common-gain-only ablations is scheduled for batch audio rendering. Select any Live Server-Room speaker on the left to audition BAFNet+ online calibration in real acoustic noise.",
      tab: "s1"
    }
  };

  let clips = [];
  let clipMap = {};
  let liveIds = [];

  const state = {
    scenId: "",
    clipId: "",
    leftSys: "acs",
    rightSys: "enhanced",
    auditionSys: "enhanced",
    wipePct: 50,
    autoWipeAudio: true,
    loop: true,
    playing: false,
    loading: false,
    offset: 0,
    tableTab: "s1",
    s1Metric: "PESQ"
  };

  const audioEng = {
    ctx: null,
    decoded: {},
    decodingPromise: {},
    nodes: null,
    rafId: null
  };

  let lastRenderedSys = null;

  function timeFracToImgXPct(f) {
    const clamped = Math.max(0, Math.min(1, f));
    return (PLOT_X0 + clamped * PLOT_SPAN_X) * 100;
  }

  function ensureContext() {
    if (!audioEng.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      audioEng.ctx = new AC();
    }
    if (audioEng.ctx.state === "suspended") {
      audioEng.ctx.resume();
    }
    return audioEng.ctx;
  }

  async function loadClipBuffers(clipId) {
    if (audioEng.decoded[clipId]) return audioEng.decoded[clipId];
    if (audioEng.decodingPromise[clipId]) return audioEng.decodingPromise[clipId];
    const ctx = ensureContext();
    const clip = clipMap[clipId];
    const keys = Object.keys(clip.tracks);
    audioEng.decodingPromise[clipId] = Promise.all(
      keys.map(k =>
        fetch(clip.tracks[k].audioUrl)
          .then(res => {
            if (!res.ok) throw new Error(`Failed to fetch ${clip.tracks[k].audioUrl} (${res.status})`);
            return res.arrayBuffer();
          })
          .then(arr => ctx.decodeAudioData(arr))
          .then(buf => [k, buf])
      )
    ).then(pairs => {
      const map = Object.fromEntries(pairs);
      audioEng.decoded[clipId] = map;
      return map;
    });
    return audioEng.decodingPromise[clipId];
  }

  function effectiveAudioSys(posFrac) {
    if (state.autoWipeAudio) {
      const curtainFrac = state.wipePct / 100;
      return posFrac < curtainFrac ? state.leftSys : state.rightSys;
    }
    return state.auditionSys;
  }

  function applyLiveGains(posFrac) {
    if (!audioEng.nodes || !audioEng.ctx) return;
    const targetSys = effectiveAudioSys(posFrac);
    const now = audioEng.ctx.currentTime;
    for (const [k, gNode] of Object.entries(audioEng.nodes.gains)) {
      gNode.gain.setTargetAtTime(k === targetSys ? 1 : 0, now, 0.012);
    }
  }

  function stopAudio(resetOffset) {
    if (audioEng.nodes) {
      if (audioEng.ctx && state.playing) {
        const elapsed = audioEng.ctx.currentTime - audioEng.nodes.startWall;
        state.offset = resetOffset ? 0 : Math.min(audioEng.nodes.duration, audioEng.nodes.startOffset + elapsed);
      }
      for (const src of Object.values(audioEng.nodes.sources)) {
        try { src.onended = null; src.stop(); src.disconnect(); } catch (_) {}
      }
      for (const g of Object.values(audioEng.nodes.gains)) {
        try { g.disconnect(); } catch (_) {}
      }
      audioEng.nodes = null;
    } else if (resetOffset) {
      state.offset = 0;
    }
    state.playing = false;
    state.loading = false;
    if (audioEng.rafId) {
      cancelAnimationFrame(audioEng.rafId);
      audioEng.rafId = null;
    }
    syncPlayUI();
    syncProgressDOM();
  }

  async function startAudio() {
    const targetClip = state.clipId;
    if (!targetClip || !clipMap[targetClip]) return;
    state.loading = true;
    syncPlayUI();

    try {
      const bufs = await loadClipBuffers(targetClip);
      if (state.clipId !== targetClip) return;

      const ctx = ensureContext();
      const clip = clipMap[targetClip];
      const dur = Object.values(bufs)[0].duration || clip.duration;
      if (state.offset >= dur - 0.05) state.offset = 0;

      const startOffset = state.offset;
      const posFrac = Math.max(0, Math.min(1, startOffset / Math.max(0.01, dur)));
      const activeAtStart = effectiveAudioSys(posFrac);

      const sources = {};
      const gains = {};
      for (const k of Object.keys(bufs)) {
        const src = ctx.createBufferSource();
        src.buffer = bufs[k];
        const gain = ctx.createGain();
        gain.gain.value = (k === activeAtStart) ? 1 : 0;
        src.connect(gain);
        gain.connect(ctx.destination);
        sources[k] = src;
        gains[k] = gain;
      }

      const startWall = ctx.currentTime + 0.025;
      audioEng.nodes = { clipId: targetClip, sources, gains, startWall, startOffset, duration: dur };
      state.loading = false;
      state.playing = true;
      for (const k of Object.keys(sources)) {
        sources[k].start(startWall, startOffset);
      }

      sources[Object.keys(sources)[0]].onended = () => {
        if (!state.playing || !audioEng.nodes || audioEng.nodes.clipId !== targetClip) return;
        const shouldLoop = state.loop;
        stopAudio(true);
        if (shouldLoop) startAudio();
      };

      syncPlayUI();
      tickRAF();
    } catch (err) {
      state.loading = false;
      state.playing = false;
      syncPlayUI();
      throw err;
    }
  }

  function togglePlay() {
    if (state.playing) {
      stopAudio(false);
    } else if (!state.loading) {
      startAudio();
    }
  }

  function seekToFraction(frac) {
    if (!clipMap[state.clipId]) return;
    const clamped = Math.max(0, Math.min(1, frac));
    const dur = clipMap[state.clipId].duration;
    state.offset = Math.max(0, Math.min(dur - 0.05, clamped * dur));
    syncProgressDOM();
    syncLiveSideBadge(clamped);
    if (state.playing) {
      stopAudio(false);
      startAudio();
    }
  }

  function currentPosSec() {
    const clip = clipMap[state.clipId];
    if (!clip) return 0;
    if (!state.playing || !audioEng.nodes || !audioEng.ctx) return Math.min(state.offset, clip.duration);
    const elapsed = Math.max(0, audioEng.ctx.currentTime - audioEng.nodes.startWall);
    return Math.min(audioEng.nodes.duration, audioEng.nodes.startOffset + elapsed);
  }

  function fmtTime(sec) {
    const s = Math.max(0, sec);
    const mins = Math.floor(s / 60);
    const rem = (s - mins * 60).toFixed(1).padStart(4, "0");
    return String(mins).padStart(2, "0") + ":" + rem;
  }

  function syncAuditionButtonsOnly(highlightedSys) {
    document.querySelectorAll(".b-sys-btn").forEach(btn => {
      const s = btn.getAttribute("data-sys");
      const isOn = s === highlightedSys;
      btn.style.background = isOn ? "#00e5dc" : "#22272a";
      btn.style.color = isOn ? "#0b1718" : "#c7d3d5";
      btn.style.border = isOn ? "1px solid #00e5dc" : "1px solid rgba(255,255,255,0.1)";

      const badge = btn.querySelector(".role-badge");
      if (badge) {
        const tags = [];
        if (state.leftSys === s) tags.push("A");
        if (state.rightSys === s) tags.push("B");
        badge.textContent = tags.length ? "[" + tags.join("/") + "]" : "";
        badge.style.color = isOn ? "#0b1718" : (tags.includes("B") ? "#00e5dc" : "#e6b980");
      }
    });
  }

  function syncLiveSideBadge(posFrac) {
    const badge = document.getElementById("b-live-side-badge");
    if (!badge) return;
    const sysKey = effectiveAudioSys(posFrac);
    if (state.autoWipeAudio) {
      const isLeft = posFrac < (state.wipePct / 100);
      badge.textContent = `HEARING: ${isLeft ? "A" : "B"} \u00b7 ${STREAM_META[sysKey].short}`;
      badge.style.background = isLeft ? "rgba(230,185,128,0.16)" : "rgba(0,229,220,0.16)";
      badge.style.color = isLeft ? "#e6b980" : "#00e5dc";
      badge.style.border = isLeft ? "1px solid rgba(230,185,128,0.45)" : "1px solid rgba(0,229,220,0.45)";
    } else {
      badge.textContent = `SOLO \u00b7 ${STREAM_META[sysKey].short}`;
      badge.style.background = "#101315";
      badge.style.color = "#d2dcde";
      badge.style.border = "1px solid rgba(255,255,255,0.18)";
    }
    syncAuditionButtonsOnly(sysKey);
  }

  function syncProgressDOM() {
    const clip = clipMap[state.clipId];
    if (!clip) return;
    const dur = audioEng.nodes ? audioEng.nodes.duration : clip.duration;
    const cur = currentPosSec();
    const posFrac = Math.max(0, Math.min(1, cur / Math.max(0.01, dur)));

    const playhead = document.getElementById("b-playhead");
    if (playhead) playhead.style.left = timeFracToImgXPct(posFrac).toFixed(3) + "%";
    const tc = document.getElementById("b-timecode");
    if (tc) tc.textContent = `${fmtTime(cur)} / ${fmtTime(dur)}`;

    const targetSys = effectiveAudioSys(posFrac);
    if (state.playing && state.autoWipeAudio && targetSys !== lastRenderedSys) {
      lastRenderedSys = targetSys;
      applyLiveGains(posFrac);
      syncLiveSideBadge(posFrac);
    }
  }

  function tickRAF() {
    if (!state.playing) return;
    syncProgressDOM();
    audioEng.rafId = requestAnimationFrame(tickRAF);
  }

  function syncPlayUI() {
    const btn = document.getElementById("b-play-btn");
    if (btn) {
      btn.textContent = state.loading ? "Loading..." : (state.playing ? "\u275a\u275a Pause" : "\u25b6 Play");
    }
    const loopBtn = document.getElementById("b-loop-btn");
    if (loopBtn) {
      loopBtn.style.background = state.loop ? "rgba(0,229,220,0.14)" : "#15181a";
      loopBtn.style.color = state.loop ? "#00e5dc" : "#8ea1a5";
      loopBtn.style.border = state.loop ? "1px solid rgba(0,229,220,0.45)" : "1px solid rgba(255,255,255,0.12)";
    }
  }

  function syncVisuals() {
    const clip = clipMap[state.clipId];
    if (!clip) return;
    const wipeImgX = timeFracToImgXPct(state.wipePct / 100);
    const baseEl = document.getElementById("b-img-base");
    const wipeEl = document.getElementById("b-wipe-overlay");
    baseEl.src = clip.tracks[state.leftSys].specUrl;
    wipeEl.src = clip.tracks[state.rightSys].specUrl;
    wipeEl.style.clipPath = `inset(0 0 0 ${wipeImgX.toFixed(3)}%)`;

    const wipeLine = document.getElementById("b-wipe-line");
    wipeLine.style.display = "block";
    wipeLine.style.left = wipeImgX.toFixed(3) + "%";

    document.getElementById("b-left-select").value = state.leftSys;
    document.getElementById("b-right-select").value = state.rightSys;
    document.getElementById("b-wipe-slider").value = state.wipePct;
    const splitSec = (clip.duration * state.wipePct / 100).toFixed(2);
    document.getElementById("b-wipe-pct").textContent = `Split @ ${splitSec} s (${state.wipePct}%)`;
    document.getElementById("b-label-left").textContent = "A \u00b7 " + STREAM_META[state.leftSys].short;
    document.getElementById("b-label-right").textContent = "B \u00b7 " + STREAM_META[state.rightSys].short;

    const autoBtn = document.getElementById("b-auto-wipe-audio");
    if (state.autoWipeAudio) {
      autoBtn.textContent = "Split Audio: ON (A \u2192 B)";
      autoBtn.style.background = "rgba(0,229,220,0.18)";
      autoBtn.style.color = "#00e5dc";
      autoBtn.style.border = "1px solid #00e5dc";
    } else {
      autoBtn.textContent = "Split Audio: SOLO (Click to Sync)";
      autoBtn.style.background = "#101315";
      autoBtn.style.color = "#a8b7ba";
      autoBtn.style.border = "1px solid rgba(255,255,255,0.16)";
    }

    document.querySelectorAll(".b-scen-btn").forEach(btn => {
      const active = btn.getAttribute("data-b-scen") === state.scenId;
      btn.classList.toggle("is-active", active);
    });

    const posFrac = currentPosSec() / Math.max(0.01, clip.duration);
    lastRenderedSys = effectiveAudioSys(posFrac);
    syncLiveSideBadge(posFrac);
    applyLiveGains(posFrac);
    syncProgressDOM();
  }

  function selectClip(clipId) {
    if (!clipMap[clipId]) return;
    const wasPlaying = state.playing;
    const changedClip = state.clipId !== clipId;
    state.scenId = clipId;
    document.getElementById("b-live-bench").classList.remove("hidden");
    document.getElementById("b-info-panel").classList.add("hidden");
    if (changedClip) {
      stopAudio(true);
      state.clipId = clipId;
    }
    syncVisuals();
    if (wasPlaying && changedClip) startAudio();
  }

  function setAuditionSys(sysKey, manualSolo = true) {
    if (manualSolo) {
      state.autoWipeAudio = false;
    }
    state.auditionSys = sysKey;
    syncVisuals();
  }

  function setWipePct(pct, engageSplitAudio = true) {
    state.wipePct = Math.max(0, Math.min(100, Math.round(pct)));
    if (engageSplitAudio) state.autoWipeAudio = true;
    syncVisuals();
  }

  function fmtNum(v, bold) {
    const txt = Number(v).toFixed(3);
    return bold ? `<strong>${txt}</strong>` : txt;
  }

  function renderPaperTable() {
    document.querySelectorAll(".t-tab").forEach(btn => {
      const active = btn.getAttribute("data-table-tab") === state.tableTab;
      btn.style.color = active ? "#006f6a" : "#4b5a5f";
      btn.style.borderBottom = active ? "2px solid #008e88" : "2px solid transparent";
    });

    const subWrap = document.getElementById("t-subfilter-wrap");
    const bodyEl = document.getElementById("t-body");

    if (state.tableTab === "s1") {
      const metrics = Object.keys(TABLE_DATA.s1.metrics);
      subWrap.innerHTML = `<span style="font-weight:500; color:#5c6c70; margin-right:4px;">Metric:</span>` + metrics.map(m => {
        const on = state.s1Metric === m;
        return `<button type="button" data-s1-metric="${m}" class="metric-btn" style="border:1px solid ${on ? "#008e88" : "#cbd4d8"}; background:${on ? "#008e88" : "#f6f8f9"}; color:${on ? "#ffffff" : "#37454a"};">${m}</button>`;
      }).join("");
      subWrap.querySelectorAll("[data-s1-metric]").forEach(b => {
        b.addEventListener("click", () => { state.s1Metric = b.getAttribute("data-s1-metric"); renderPaperTable(); });
      });

      const block = TABLE_DATA.s1.metrics[state.s1Metric];
      bodyEl.innerHTML = `
        <table class="ieee-table">
          <thead>
            <tr>
              <th>Model</th>
              ${block.snrs.map(s => `<th class="num">${s}</th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${block.rows.map(r => {
              const cls = r.highlight ? "row-highlight row-bottom" : "";
              return `<tr class="${cls}">
                <td>${r.model}</td>
                ${r.vals.map((v, i) => `<td class="num">${fmtNum(v, r.bold[i])}</td>`).join("")}
              </tr>`;
            }).join("")}
          </tbody>
        </table>`;
    } else if (state.tableTab === "s5") {
      subWrap.innerHTML = `<span style="font-weight:500; color:#5c6c70;">Cross-Corpus PESQ (N = 3,064 &times; 2)</span>`;
      const cols = ["\u221220 dB", "\u221210 dB", "0 dB", "+10 dB", "+15 dB"];
      bodyEl.innerHTML = `
        <table class="ieee-table">
          <thead>
            <tr>
              <th>Model / Branch</th>
              ${cols.map(c => `<th class="num">${c}</th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${TABLE_DATA.s5.tiers.map((tier, tIdx) => `
              <tr class="row-section"><td colspan="6">${tier.title}</td></tr>
              ${tier.rows.map((r, rIdx) => {
                const isLastOverall = tIdx === TABLE_DATA.s5.tiers.length - 1 && rIdx === tier.rows.length - 1;
                const cls = [r.highlight ? "row-highlight" : "", isLastOverall ? "row-bottom" : ""].filter(Boolean).join(" ");
                return `<tr class="${cls}">
                  <td>${r.model}</td>
                  ${r.vals.map((v, i) => `<td class="num">${fmtNum(v, r.bold[i])}</td>`).join("")}
                </tr>`;
              }).join("")}
            `).join("")}
          </tbody>
        </table>`;
    }
  }

  function renderRailClips(sec3Items) {
    const railEl = document.getElementById("b-rail-live");
    const countEl = document.getElementById("b-rail-count");
    clips = sec3Items
      .filter(it => it.clips && it.clips.enhanced && it.clips.enhanced.status === "ready")
      .map(it => {
        const tracks = {};
        for (const k of ["acs", "bcs", "earphone", "enhanced"]) {
          const c = it.clips[k];
          tracks[k] = {
            label: c.label,
            audioUrl: c.audio_url,
            specUrl: c.spectrogram_url
          };
        }
        return {
          id: it.id,
          shortTitle: `${it.speaker_label} \u00b7 ${it.sentence_id.toUpperCase()}`,
          subtitle: `${it.sentence_label} \u00b7 ${Number(it.duration_sec).toFixed(2)} s \u00b7 ${it.gender === "F" ? "Female" : "Male"}`,
          duration: Number(it.duration_sec),
          tracks
        };
      });

    clipMap = Object.fromEntries(clips.map(c => [c.id, c]));
    liveIds = clips.map(c => c.id);
    if (countEl) countEl.textContent = `${clips.length}/${sec3Items.length} Live`;

    railEl.innerHTML = clips.map(c => `
      <button type="button" data-b-scen="${c.id}" class="b-scen-btn">
        <div class="b-scen-meta">
          <div class="b-scen-title">${c.shortTitle}</div>
          <div class="b-scen-sub">${c.subtitle}</div>
        </div>
        <span class="rail-pill-live">LIVE</span>
      </button>
    `).join("");

    document.querySelectorAll(".b-scen-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-b-scen");
        if (clipMap[key]) {
          selectClip(key);
        } else if (QUEUED_INFO[key]) {
          if (state.playing) stopAudio(false);
          state.scenId = key;
          document.getElementById("b-live-bench").classList.add("hidden");
          document.getElementById("b-info-panel").classList.remove("hidden");
          const info = QUEUED_INFO[key];
          document.getElementById("b-info-title").textContent = info.title;
          document.getElementById("b-info-desc").textContent = info.desc;
          state.tableTab = info.tab;
          renderPaperTable();
          syncVisuals();
        }
      });
    });

    if (liveIds.length > 0) {
      state.scenId = liveIds[0];
      state.clipId = liveIds[0];
    }
  }

  function wireStaticControls() {
    document.getElementById("b-back-live").addEventListener("click", () => selectClip(state.clipId));

    document.getElementById("b-left-select").addEventListener("change", e => {
      state.leftSys = e.target.value;
      state.autoWipeAudio = true;
      syncVisuals();
    });
    document.getElementById("b-right-select").addEventListener("change", e => {
      state.rightSys = e.target.value;
      state.autoWipeAudio = true;
      syncVisuals();
    });
    document.getElementById("b-swap-ab").addEventListener("click", () => {
      const tmp = state.leftSys;
      state.leftSys = state.rightSys;
      state.rightSys = tmp;
      state.autoWipeAudio = true;
      syncVisuals();
    });
    document.getElementById("b-auto-wipe-audio").addEventListener("click", () => {
      state.autoWipeAudio = !state.autoWipeAudio;
      lastRenderedSys = null;
      syncVisuals();
    });

    document.getElementById("b-wipe-slider").addEventListener("input", e => setWipePct(Number(e.target.value), true));
    document.querySelectorAll("[data-wipe-preset]").forEach(btn => {
      btn.addEventListener("click", () => setWipePct(Number(btn.getAttribute("data-wipe-preset")), true));
    });

    let draggingWipe = false;
    const wipeHandle = document.getElementById("b-wipe-handle");
    const specStage = document.getElementById("b-spec-stage");

    function clientXToTimeFrac(clientX) {
      const r = specStage.getBoundingClientRect();
      const plotLeft = r.left + r.width * PLOT_X0;
      const plotWidth = Math.max(1, r.width * PLOT_SPAN_X);
      return Math.max(0, Math.min(1, (clientX - plotLeft) / plotWidth));
    }

    wipeHandle.addEventListener("pointerdown", e => {
      e.stopPropagation();
      draggingWipe = true;
      try { wipeHandle.setPointerCapture(e.pointerId); } catch (_) {}
    });
    window.addEventListener("pointermove", e => {
      if (!draggingWipe) return;
      setWipePct(clientXToTimeFrac(e.clientX) * 100, true);
    });
    const endWipeDrag = () => { draggingWipe = false; };
    window.addEventListener("pointerup", endWipeDrag);
    window.addEventListener("pointercancel", endWipeDrag);
    wipeHandle.addEventListener("click", e => e.stopPropagation());

    specStage.addEventListener("click", e => {
      if (draggingWipe) return;
      seekToFraction(clientXToTimeFrac(e.clientX));
    });

    document.getElementById("b-play-btn").addEventListener("click", togglePlay);
    document.getElementById("b-loop-btn").addEventListener("click", () => {
      state.loop = !state.loop;
      syncPlayUI();
    });

    document.querySelectorAll(".b-sys-btn").forEach(btn => {
      btn.addEventListener("click", () => setAuditionSys(btn.getAttribute("data-sys"), true));
    });

    document.querySelectorAll(".t-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        state.tableTab = btn.getAttribute("data-table-tab");
        renderPaperTable();
      });
    });

    document.addEventListener("keydown", e => {
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "select" || tag === "textarea") return;

      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "1") { setAuditionSys("acs", true); }
      else if (e.key === "2") { setAuditionSys("bcs", true); }
      else if (e.key === "3") { setAuditionSys("earphone", true); }
      else if (e.key === "4") { setAuditionSys("enhanced", true); }
      else if (e.key.toLowerCase() === "s") {
        state.autoWipeAudio = true;
        syncVisuals();
      } else if (e.key.toLowerCase() === "a") { setAuditionSys(state.leftSys, true); }
      else if (e.key.toLowerCase() === "b") { setAuditionSys(state.rightSys, true); }
      else if (e.key === "[") { setWipePct(state.wipePct - 5, true); }
      else if (e.key === "]") { setWipePct(state.wipePct + 5, true); }
      else if (e.key === "ArrowLeft" && liveIds.length) {
        const idx = liveIds.indexOf(state.clipId);
        selectClip(liveIds[(idx - 1 + liveIds.length) % liveIds.length]);
      } else if (e.key === "ArrowRight" && liveIds.length) {
        const idx = liveIds.indexOf(state.clipId);
        selectClip(liveIds[(idx + 1) % liveIds.length]);
      }
    });
  }

  async function init() {
    window.__siteErrors = [];
    window.addEventListener("error", e => { window.__siteErrors.push(String(e.message || e)); });

    wireStaticControls();
    renderPaperTable();

    const res = await fetch("./samples_manifest.json");
    if (!res.ok) throw new Error(`Failed to fetch samples_manifest.json (${res.status})`);
    const manifest = await res.json();
    renderRailClips(manifest.sections.section3.items || []);
    syncPlayUI();
    syncVisuals();

    window.__benchApi = {
      state,
      loadClipBuffers,
      effectiveAudioSys,
      setWipePct,
      timeFracToImgXPct,
      selectClip,
      getClipIds: () => liveIds.slice()
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      init().catch(err => {
        window.__siteErrors = window.__siteErrors || [];
        window.__siteErrors.push(String(err));
      });
    });
  } else {
    init().catch(err => {
      window.__siteErrors = window.__siteErrors || [];
      window.__siteErrors.push(String(err));
    });
  }
})();
