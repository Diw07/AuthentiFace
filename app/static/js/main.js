/**
 * DeepFake Detector — Forensics Dashboard Frontend
 * Handles file upload, API calls, dynamic result rendering, history, and dark mode.
 */

// ─── DOM Elements ────────────────────────────────────────
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadBtnWrapper = document.getElementById('uploadBtnWrapper');
const filePreview = document.getElementById('filePreview');
const previewThumb = document.getElementById('previewThumb');
const previewVid = document.getElementById('previewVid');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const clearFileBtn = document.getElementById('clearFileBtn');
const loadingState = document.getElementById('loadingState');
const loadingStatus = document.getElementById('loadingStatus');
const systemStatus = document.getElementById('systemStatus');
const statusDot = document.getElementById('statusDot');

// Steps
const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');
const step4 = document.getElementById('step4');

// Results
const forensicsSection = document.getElementById('forensicsSection');
const framesSection = document.getElementById('framesSection');
const gradcamImg = document.getElementById('gradcamImg');
const heatmapPlaceholder = document.getElementById('heatmapPlaceholder');
const heatmapBadge = document.getElementById('heatmapBadge');
const heatmapDesc = document.getElementById('heatmapDesc');
const regionBars = document.getElementById('regionBars');
const regionBadge = document.getElementById('regionBadge');
const faceConfVal = document.getElementById('faceConfVal');
const procTimeVal = document.getElementById('procTimeVal');
const faceCropImg = document.getElementById('faceCropImg');
const faceCropPlaceholder = document.getElementById('faceCropPlaceholder');

// Gauge
const gaugeArc = document.getElementById('gaugeArc');
const gaugePercent = document.getElementById('gaugePercent');
const verdictLabel = document.getElementById('verdictLabel');
const verdictSub = document.getElementById('verdictSub');
const threatLabel = document.getElementById('threatLabel');
const threatBar = document.getElementById('threatBar');

// Frame analysis
const frameSummary = document.getElementById('frameSummary');
const frameTimeline = document.getElementById('frameTimeline');

// History
const navHistory = document.getElementById('navHistory');
const historySidebar = document.getElementById('historySidebar');
const historyOverlay = document.getElementById('historyOverlay');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const historyList = document.getElementById('historyList');

// Other
const darkModeToggle = document.getElementById('darkModeToggle');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

let selectedFile = null;

// ─── Dark Mode ───────────────────────────────────────────
if (localStorage.getItem('darkMode') === 'true' || (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
}
darkModeToggle.addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
    localStorage.setItem('darkMode', document.documentElement.classList.contains('dark'));
});

// ─── File Upload ─────────────────────────────────────────
uploadBtn.addEventListener('click', () => {
    if (selectedFile) {
        analyzeFile();
    } else {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

clearFileBtn.addEventListener('click', resetUpload);

analyzeAnotherBtn.addEventListener('click', () => {
    resetUpload();
    forensicsSection.classList.add('hidden');
    framesSection.classList.add('hidden');
    analyzeAnotherBtn.classList.add('hidden');
    resetGauge();
    resetSteps();
});

function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    filePreview.classList.remove('hidden');

    // Show thumbnail
    if (file.type.startsWith('image/')) {
        previewThumb.classList.remove('hidden');
        previewVid.classList.add('hidden');
        previewThumb.src = URL.createObjectURL(file);
    } else if (file.type.startsWith('video/')) {
        previewVid.classList.remove('hidden');
        previewThumb.classList.add('hidden');
        previewVid.src = URL.createObjectURL(file);
    }

    // Change button text
    uploadBtn.querySelector('span.relative').innerHTML = `
        <span class="material-symbols-outlined">play_arrow</span>
        Analyze Now
    `;
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    filePreview.classList.add('hidden');
    previewThumb.classList.add('hidden');
    previewVid.classList.add('hidden');
    loadingState.classList.add('hidden');
    uploadBtnWrapper.classList.remove('hidden');
    uploadBtn.querySelector('span.relative').innerHTML = `
        <span class="material-symbols-outlined">cloud_upload</span>
        Upload Media for Analysis
    `;
    systemStatus.textContent = 'System Ready';
    systemStatus.className = 'px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-xs font-semibold rounded-md border border-green-200 dark:border-green-800';
}

// ─── Analysis ────────────────────────────────────────────
async function analyzeFile() {
    if (!selectedFile) return;

    const isVideo = selectedFile.type.startsWith('video/');
    const endpoint = isVideo ? '/api/predict-video' : '/api/predict';

    // Show loading
    uploadBtnWrapper.classList.add('hidden');
    loadingState.classList.remove('hidden');
    loadingStatus.textContent = 'Detecting faces...';
    systemStatus.textContent = 'Processing...';
    systemStatus.className = 'px-2 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 text-xs font-semibold rounded-md border border-yellow-200 dark:border-yellow-800';
    statusDot.className = 'w-2 h-2 rounded-full bg-yellow-500 animate-pulse';
    activateStep(2);

    // Simulate status updates
    const statusMessages = isVideo
        ? ['Extracting frames...', 'Running face detection...', 'Classifying frames...', 'Aggregating results...']
        : ['Cropping face region...', 'Running ConvNeXt-Tiny...', 'Generating Grad-CAM...', 'Computing region scores...'];
    let msgIdx = 0;
    const statusInterval = setInterval(() => {
        if (msgIdx < statusMessages.length) {
            loadingStatus.textContent = statusMessages[msgIdx++];
        }
    }, 800);

    const startTime = performance.now();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(endpoint, { method: 'POST', body: formData });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const result = await response.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        clearInterval(statusInterval);
        loadingState.classList.add('hidden');

        // Update system status
        systemStatus.textContent = 'Analysis Complete';
        systemStatus.className = 'px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-xs font-semibold rounded-md border border-green-200 dark:border-green-800';
        statusDot.className = 'w-2 h-2 rounded-full bg-green-500 shadow-glow-success animate-pulse';
        activateStep(3);

        // Display results
        // Normalize: video uses 'verdict', image uses 'label'
        const normalizedLabel = result.label || result.verdict || 'UNKNOWN';

        // Handle no-face-detected
        if (!isVideo && result.face_detected === false) {
            displayNoFaceResult(result, elapsed);
        } else if (isVideo) {
            displayVideoResult(result, elapsed);
        } else {
            displayImageResult(result, elapsed);
        }

        activateStep(4);
        analyzeAnotherBtn.classList.remove('hidden');
        saveToHistory(selectedFile.name, { ...result, label: normalizedLabel }, isVideo);

    } catch (err) {
        clearInterval(statusInterval);
        loadingState.classList.add('hidden');
        uploadBtnWrapper.classList.remove('hidden');
        systemStatus.textContent = 'Error';
        systemStatus.className = 'px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-xs font-semibold rounded-md border border-red-200 dark:border-red-800';
        statusDot.className = 'w-2 h-2 rounded-full bg-red-500';
        alert('Error: ' + err.message);
    }
}

// ─── Display No-Face Result ──────────────────────────────
function displayNoFaceResult(result, elapsed) {
    forensicsSection.classList.remove('hidden');
    framesSection.classList.add('hidden');

    procTimeVal.textContent = elapsed + 's';
    faceConfVal.textContent = 'Not found';

    // Gauge shows warning state
    gaugeArc.style.strokeDashoffset = 408.4;
    gaugeArc.style.stroke = '#F59E0B';
    gaugePercent.textContent = '?';
    verdictLabel.textContent = 'NO FACE';
    verdictLabel.className = 'text-2xl font-black text-amber-500';
    verdictSub.textContent = result.error || 'No face detected in image';
    threatBar.style.width = '0%';
    threatLabel.textContent = 'N/A';
    threatLabel.className = 'font-bold text-amber-500';

    // Heatmap card
    heatmapBadge.textContent = 'No Face';
    heatmapBadge.className = 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 text-[10px] px-2 py-0.5 rounded border border-amber-200 dark:border-amber-800';
    heatmapDesc.innerHTML = '<span class="text-amber-500 font-bold">No face detected.</span> Try uploading a clearer image with a visible face. The face detector (RetinaFace) needs a clear frontal or near-frontal face.';

    // Region card
    regionBadge.textContent = 'N/A';
    regionBadge.className = 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 text-[10px] px-2 py-0.5 rounded border border-amber-200 dark:border-amber-800';
    regionBars.innerHTML = '<p class="text-xs text-amber-400 py-8 text-center">No face detected — region analysis unavailable</p>';

    // System status
    systemStatus.textContent = 'No Face Found';
    systemStatus.className = 'px-2 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 text-xs font-semibold rounded-md border border-amber-200 dark:border-amber-800';
}

// ─── Display Image Result ────────────────────────────────
function displayImageResult(result, elapsed) {
    forensicsSection.classList.remove('hidden');
    framesSection.classList.add('hidden');

    // Gauge
    const confidence = result.confidence || 0;
    const isFake = result.label === 'FAKE';
    setGauge(confidence, isFake);

    // Processing time
    procTimeVal.textContent = elapsed + 's';

    // Face confidence
    if (result.detection_confidence) {
        faceConfVal.textContent = (result.detection_confidence * 100).toFixed(1) + '%';
    }

    // Grad-CAM
    if (result.gradcam_image) {
        gradcamImg.src = 'data:image/png;base64,' + result.gradcam_image;
        gradcamImg.classList.remove('hidden');
        heatmapPlaceholder.classList.add('hidden');
        heatmapBadge.textContent = isFake ? 'Critical' : 'Low';
        heatmapBadge.className = isFake
            ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 text-[10px] px-2 py-0.5 rounded border border-red-200 dark:border-red-800'
            : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-[10px] px-2 py-0.5 rounded border border-green-200 dark:border-green-800';
        heatmapDesc.innerHTML = isFake
            ? `<span class="text-red-500 font-bold">${(confidence * 100).toFixed(0)}% Confidence:</span> High activation detected in facial regions suggesting manipulation artifacts.`
            : `<span class="text-green-500 font-bold">${((1 - confidence) * 100).toFixed(0)}% Confidence:</span> No significant manipulation artifacts detected.`;

        // Also show as face crop
        faceCropImg.src = gradcamImg.src;
        faceCropImg.classList.remove('hidden');
        faceCropPlaceholder.classList.add('hidden');
    }

    // Region Scores
    if (result.region_scores) {
        const sorted = Object.entries(result.region_scores).sort((a, b) => b[1] - a[1]);
        const maxScore = Math.max(...sorted.map(s => s[1]));
        regionBadge.textContent = maxScore >= 50 ? 'Warning' : 'Normal';
        regionBadge.className = maxScore >= 50
            ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 text-[10px] px-2 py-0.5 rounded border border-orange-200 dark:border-orange-800'
            : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-[10px] px-2 py-0.5 rounded border border-green-200 dark:border-green-800';

        regionBars.innerHTML = sorted.map(([name, score]) => {
            const color = score >= 60 ? '#EF4444' : score >= 30 ? '#F59E0B' : '#10B981';
            return `
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-slate-500 dark:text-slate-400">${name}</span>
                        <span class="font-bold" style="color: ${color}">${score.toFixed(1)}%</span>
                    </div>
                    <div class="h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div class="h-full rounded-full transition-all duration-1000" style="width: 0%; background: ${color};" data-target-width="${score}%"></div>
                    </div>
                </div>
            `;
        }).join('');

        // Animate bars
        setTimeout(() => {
            regionBars.querySelectorAll('[data-target-width]').forEach(el => {
                el.style.width = el.getAttribute('data-target-width');
            });
        }, 100);
    }
}

// ─── Display Video Result ────────────────────────────────
function displayVideoResult(result, elapsed) {
    forensicsSection.classList.remove('hidden');
    framesSection.classList.remove('hidden');

    // Gauge — video uses 'verdict' not 'label'
    const confidence = result.confidence || 0;
    const verdict = result.verdict || result.label || 'UNKNOWN';
    const isFake = verdict === 'FAKE';
    setGauge(confidence, isFake);

    procTimeVal.textContent = elapsed + 's';

    // Heatmap card for video
    heatmapBadge.textContent = 'Video';
    heatmapBadge.className = 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 text-[10px] px-2 py-0.5 rounded border border-blue-200 dark:border-blue-800';
    heatmapDesc.innerHTML = isFake
        ? `<span class="text-red-500 font-bold">${(confidence * 100).toFixed(0)}% Confidence:</span> Multiple frames show manipulation artifacts. Click frames below to inspect.`
        : `<span class="text-green-500 font-bold">${((1 - confidence) * 100).toFixed(0)}% Confidence:</span> Frame-by-frame analysis shows no manipulation. Click frames below.`;

    // Frame summary — video uses 'per_frame_results' not 'frame_results'
    const frames = result.per_frame_results || result.frame_results || [];
    const fakeCount = frames.filter(f => f.label === 'FAKE').length;
    const realCount = frames.filter(f => f.label === 'REAL').length;
    const noFace = frames.filter(f => f.label === 'UNKNOWN' || !f.face_detected).length;

    frameSummary.innerHTML = `
        <span class="px-3 py-1.5 rounded-lg text-xs font-bold ${isFake ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'}">
            ${verdict} (${(confidence * 100).toFixed(1)}%)
        </span>
        <span class="text-xs text-slate-500 dark:text-slate-400">${frames.length} frames analyzed</span>
        <span class="text-xs text-red-500">${fakeCount} fake</span>
        <span class="text-xs text-green-500">${realCount} real</span>
        ${noFace > 0 ? `<span class="text-xs text-slate-400">${noFace} no face</span>` : ''}
    `;

    // Frame dots — clickable to view each frame
    frameTimeline.innerHTML = frames.map((f, i) => {
        let dotClass, tooltip;
        if (f.label === 'FAKE') {
            dotClass = 'bg-red-500/20 border-red-500/50 text-red-400 hover:bg-red-500/40';
            tooltip = `Frame ${i + 1}: FAKE (${(f.confidence * 100).toFixed(0)}%)`;
        } else if (f.label === 'REAL') {
            dotClass = 'bg-green-500/20 border-green-500/50 text-green-400 hover:bg-green-500/40';
            tooltip = `Frame ${i + 1}: REAL (${((1 - f.confidence) * 100).toFixed(0)}%)`;
        } else {
            dotClass = 'bg-slate-500/10 border-slate-500/30 text-slate-400 hover:bg-slate-500/20';
            tooltip = `Frame ${i + 1}: No face detected`;
        }
        return `<div class="w-10 h-10 rounded-lg border text-[10px] font-bold flex items-center justify-center ${dotClass} cursor-pointer transition-all" title="${tooltip}" data-frame-idx="${i}">${i + 1}</div>`;
    }).join('');

    // Click handler for frame dots
    frameTimeline.querySelectorAll('[data-frame-idx]').forEach(dot => {
        dot.addEventListener('click', () => {
            const idx = parseInt(dot.getAttribute('data-frame-idx'));
            showVideoFrame(frames[idx], idx);
        });
    });

    // Show first frame with highest fake score, or first frame
    const bestFrame = frames.reduce((best, f, i) => {
        if (!best || f.confidence > best.confidence) return { ...f, _idx: i };
        return best;
    }, null);
    if (bestFrame) showVideoFrame(bestFrame, bestFrame._idx || 0);

    // Aggregate region scores for video (from backend)
    if (result.region_scores) {
        const sorted = Object.entries(result.region_scores).sort((a, b) => b[1] - a[1]);
        const maxScore = Math.max(...sorted.map(s => s[1]));
        regionBadge.textContent = maxScore >= 50 ? 'Warning' : 'Normal';
        regionBadge.className = maxScore >= 50
            ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 text-[10px] px-2 py-0.5 rounded border border-orange-200 dark:border-orange-800'
            : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-[10px] px-2 py-0.5 rounded border border-green-200 dark:border-green-800';

        regionBars.innerHTML = sorted.map(([name, score]) => {
            const color = score >= 60 ? '#EF4444' : score >= 30 ? '#F59E0B' : '#10B981';
            return `
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-slate-500 dark:text-slate-400">${name}</span>
                        <span class="font-bold" style="color: ${color}">${score.toFixed(1)}%</span>
                    </div>
                    <div class="h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div class="h-full rounded-full transition-all duration-1000" style="width: 0%; background: ${color};" data-target-width="${score}%"></div>
                    </div>
                </div>
            `;
        }).join('');

        setTimeout(() => {
            regionBars.querySelectorAll('[data-target-width]').forEach(el => {
                el.style.width = el.getAttribute('data-target-width');
            });
        }, 100);
    } else {
        regionBadge.textContent = 'Pending';
        regionBars.innerHTML = '<p class="text-xs text-slate-400 py-4 text-center">Region analysis across frames</p>';
    }
}

// ─── Show a specific video frame ─────────────────────────
function showVideoFrame(frame, idx) {
    // Show the analyzed frame in heatmap area
    if (frame.gradcam_image) {
        gradcamImg.src = 'data:image/png;base64,' + frame.gradcam_image;
        gradcamImg.classList.remove('hidden');
        heatmapPlaceholder.classList.add('hidden');
    } else if (frame.frame_image) {
        gradcamImg.src = 'data:image/png;base64,' + frame.frame_image;
        gradcamImg.classList.remove('hidden');
        heatmapPlaceholder.classList.add('hidden');
    }

    // Show frame image in face crop area
    if (frame.frame_image) {
        faceCropImg.src = 'data:image/png;base64,' + frame.frame_image;
        faceCropImg.classList.remove('hidden');
        faceCropPlaceholder.classList.add('hidden');
    }

    // Update heatmap description for this frame
    const frameFake = frame.label === 'FAKE';
    const conf = frame.confidence || 0;
    heatmapDesc.innerHTML = frameFake
        ? `<span class="text-red-500 font-bold">Frame ${idx + 1}:</span> ${(conf * 100).toFixed(0)}% fake probability — manipulation artifacts detected.`
        : `<span class="text-green-500 font-bold">Frame ${idx + 1}:</span> ${((1 - conf) * 100).toFixed(0)}% real probability — no significant artifacts.`;

    // Highlight active frame dot
    frameTimeline.querySelectorAll('[data-frame-idx]').forEach(dot => {
        const isActive = parseInt(dot.getAttribute('data-frame-idx')) === idx;
        dot.classList.toggle('ring-2', isActive);
        dot.classList.toggle('ring-blue-500', isActive);
        dot.classList.toggle('scale-110', isActive);
    });
}

// ─── Gauge ───────────────────────────────────────────────
function setGauge(confidence, isFake) {
    const circumference = 408.4;
    const pct = confidence * 100;
    const offset = circumference - (pct / 100) * circumference;

    gaugeArc.style.strokeDashoffset = offset;
    gaugeArc.style.stroke = isFake ? '#EF4444' : '#10B981';
    animateValue(gaugePercent, 0, pct, 1200, '%');

    verdictLabel.textContent = isFake ? 'FAKE' : 'REAL';
    verdictLabel.className = `text-2xl font-black ${isFake ? 'text-red-500' : 'text-green-500'}`;
    verdictSub.textContent = isFake ? 'Manipulation detected' : 'No manipulation detected';

    // Threat bar
    threatBar.style.width = pct + '%';
    threatBar.style.background = isFake ? '#EF4444' : '#10B981';
    if (pct >= 70) {
        threatLabel.textContent = 'High';
        threatLabel.className = 'font-bold text-red-500';
    } else if (pct >= 40) {
        threatLabel.textContent = 'Moderate';
        threatLabel.className = 'font-bold text-orange-500';
    } else {
        threatLabel.textContent = 'Low';
        threatLabel.className = 'font-bold text-green-500';
    }
}

function resetGauge() {
    gaugeArc.style.strokeDashoffset = 408.4;
    gaugeArc.style.stroke = '#94a3b8';
    gaugePercent.textContent = '—';
    verdictLabel.textContent = 'AWAITING';
    verdictLabel.className = 'text-2xl font-black text-slate-400';
    verdictSub.textContent = 'Upload media to begin';
    threatBar.style.width = '0%';
    threatLabel.textContent = 'None';
    threatLabel.className = 'font-bold text-slate-400';
    faceConfVal.textContent = '—';
    procTimeVal.textContent = '—';

    // Reset heatmap
    gradcamImg.classList.add('hidden');
    heatmapPlaceholder.classList.remove('hidden');
    heatmapPlaceholder.textContent = 'Upload media to view heatmap';
    heatmapBadge.textContent = 'Pending';
    heatmapBadge.className = 'text-[10px] px-2 py-0.5 rounded border bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-700';
    heatmapDesc.textContent = 'Grad-CAM highlights regions that most influenced the deepfake prediction.';
    regionBars.innerHTML = '<p class="text-xs text-slate-400 py-8 text-center">Face regions will appear after analysis</p>';
    regionBadge.textContent = 'Pending';
    regionBadge.className = 'text-[10px] px-2 py-0.5 rounded border bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-700';
    faceCropImg.classList.add('hidden');
    faceCropPlaceholder.classList.remove('hidden');
}

// ─── Steps ───────────────────────────────────────────────
function activateStep(num) {
    const steps = [step1, step2, step3, step4];
    const colors = ['blue', 'indigo', 'purple', 'emerald'];
    steps.forEach((step, i) => {
        const icon = step.querySelector('div');
        if (i < num) {
            icon.className = `w-16 h-16 rounded-2xl bg-white dark:bg-slate-800 shadow-lg border-2 border-${colors[i]}-400 dark:border-${colors[i]}-500 flex items-center justify-center text-${colors[i]}-500 transition-all duration-300`;
        } else {
            icon.className = 'w-16 h-16 rounded-2xl bg-white dark:bg-slate-800 shadow-lg border border-slate-200 dark:border-slate-600 flex items-center justify-center text-slate-400 transition-all duration-300';
        }
    });
}

function resetSteps() {
    activateStep(1);
}

// ─── Animate Value ───────────────────────────────────────
function animateValue(el, start, end, duration, suffix = '') {
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);
        el.textContent = current + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ─── History ─────────────────────────────────────────────
navHistory.addEventListener('click', () => {
    historySidebar.classList.toggle('translate-x-full');
    historySidebar.classList.toggle('translate-x-0');
    historyOverlay.classList.toggle('hidden');
});
closeHistoryBtn.addEventListener('click', closeHistory);
historyOverlay.addEventListener('click', closeHistory);
clearHistoryBtn.addEventListener('click', () => {
    localStorage.removeItem('dfHistory');
    renderHistory();
});

function closeHistory() {
    historySidebar.classList.add('translate-x-full');
    historySidebar.classList.remove('translate-x-0');
    historyOverlay.classList.add('hidden');
}

function saveToHistory(filename, result, isVideo) {
    const history = JSON.parse(localStorage.getItem('dfHistory') || '[]');
    history.unshift({
        filename,
        label: result.label,
        confidence: result.confidence,
        isVideo,
        timestamp: Date.now(),
    });
    if (history.length > 50) history.length = 50;
    localStorage.setItem('dfHistory', JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('dfHistory') || '[]');
    if (history.length === 0) {
        historyList.innerHTML = '<p class="text-xs text-slate-400 text-center py-8">No scans yet</p>';
        return;
    }

    historyList.innerHTML = history.map(item => {
        const isFake = item.label === 'FAKE';
        const pct = (item.confidence * 100).toFixed(0);
        const time = getTimeAgo(item.timestamp);
        return `
            <div class="p-3 rounded-lg border border-slate-200 dark:border-slate-700 mb-2 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors">
                <div class="flex justify-between items-center mb-1">
                    <span class="text-xs font-semibold text-slate-700 dark:text-slate-200 truncate max-w-[180px]">${item.filename}</span>
                    <span class="text-[10px] px-2 py-0.5 rounded font-bold ${isFake ? 'bg-red-100 dark:bg-red-900/30 text-red-500' : 'bg-green-100 dark:bg-green-900/30 text-green-500'}">${item.label}</span>
                </div>
                <div class="flex justify-between text-[10px] text-slate-400">
                    <span>${item.isVideo ? '🎬 Video' : '🖼️ Image'} • ${pct}%</span>
                    <span>${time}</span>
                </div>
            </div>
        `;
    }).join('');
}

// ─── Utilities ───────────────────────────────────────────
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function getTimeAgo(ts) {
    const diff = Date.now() - ts;
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
    if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
    return Math.floor(diff / 86400000) + 'd ago';
}

// ─── Init ────────────────────────────────────────────────
renderHistory();
