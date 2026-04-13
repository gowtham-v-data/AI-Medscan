/**
 * AI MedScan — Medical Image Intelligence Platform
 * Frontend Application Logic
 * 
 * Handles: File upload, API communication, result rendering,
 * chart visualization, Grad-CAM display, and clinical report rendering.
 */

// ═══════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════

const CONFIG = {
    API_BASE: window.location.origin,
    MAX_FILE_SIZE: 50 * 1024 * 1024,
    ALLOWED_TYPES: ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'],
    ANIMATION_DELAY: 600,
    CHART_COLORS: {
        blue: '#3b82f6',
        green: '#22c55e',
        red: '#ef4444',
        orange: '#f59e0b',
        yellow: '#eab308',
        purple: '#8b5cf6',
        cyan: '#06b6d4',
        teal: '#14b8a6',
        pink: '#ec4899'
    }
};

// ═══════════════════════════════════════════
// State Management
// ═══════════════════════════════════════════

const state = {
    selectedFile: null,
    analysisResult: null,
    heatmapImages: {},
    charts: {},
    isAnalyzing: false
};

// ═══════════════════════════════════════════
// Initialize Application
// ═══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUploader();
    initTabs();
    checkServerHealth();
    initLucideIcons();
});

function initLucideIcons() {
    if (window.lucide) {
        lucide.createIcons();
    }
}

// ═══════════════════════════════════════════
// Particle Background Animation
// ═══════════════════════════════════════════

function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let particles = [];
    let animationId;
    
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    function createParticle() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            radius: Math.random() * 1.5 + 0.5,
            opacity: Math.random() * 0.4 + 0.1,
            color: Math.random() > 0.5 ? '0, 212, 255' : '0, 245, 212'
        };
    }
    
    function initializeParticles() {
        const count = Math.min(Math.floor((canvas.width * canvas.height) / 15000), 80);
        particles = Array.from({ length: count }, createParticle);
    }
    
    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach((p, i) => {
            // Update position
            p.x += p.vx;
            p.y += p.vy;
            
            // Wrap around
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
            
            // Draw particle
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${p.color}, ${p.opacity})`;
            ctx.fill();
            
            // Draw connections
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[j].x - p.x;
                const dy = particles[j].y - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 212, 255, ${0.06 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        });
        
        animationId = requestAnimationFrame(drawParticles);
    }
    
    resize();
    initializeParticles();
    drawParticles();
    
    window.addEventListener('resize', () => {
        resize();
        initializeParticles();
    });
}

// ═══════════════════════════════════════════
// Navigation
// ═══════════════════════════════════════════

function initNavigation() {
    const navbar = document.getElementById('navbar');
    
    // Scroll handling
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        // Update active nav link
        updateActiveNavLink();
    });
    
    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link, .btn-primary[href^="#"]').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
}

function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const scrollPos = window.scrollY + 200;
    
    sections.forEach(section => {
        const top = section.offsetTop;
        const height = section.offsetHeight;
        const id = section.getAttribute('id');
        const link = document.querySelector(`.nav-link[href="#${id}"]`);
        
        if (link) {
            if (scrollPos >= top && scrollPos < top + height) {
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        }
    });
}

// ═══════════════════════════════════════════
// Scroll Animations
// ═══════════════════════════════════════════

function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = entry.target.getAttribute('data-delay') || 0;
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, parseInt(delay));
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('[data-aos]').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// ═══════════════════════════════════════════
// Server Health Check
// ═══════════════════════════════════════════

async function checkServerHealth() {
    const statusDot = document.getElementById('systemStatus');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/api/health`);
        if (response.ok) {
            const data = await response.json();
            statusDot.classList.add('online');
            statusDot.classList.remove('offline');
            statusText.textContent = `Online • ${data.model_mode} mode`;
        } else {
            throw new Error('Server error');
        }
    } catch (e) {
        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
        statusText.textContent = 'Offline';
    }
    
    // Re-check every 30 seconds
    setTimeout(checkServerHealth, 30000);
}

// ═══════════════════════════════════════════
// File Upload Handler
// ═══════════════════════════════════════════

function initUploader() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const clearBtn = document.getElementById('clearImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());
    
    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Clear image
    clearBtn.addEventListener('click', clearUpload);
    
    // Analyze button
    analyzeBtn.addEventListener('click', runAnalysis);
}

function handleFile(file) {
    // Validate file type
    if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
        showNotification('Unsupported file type. Please upload JPEG, PNG, BMP, TIFF, or WebP.', 'error');
        return;
    }
    
    // Validate file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showNotification('File too large. Maximum size is 50MB.', 'error');
        return;
    }
    
    state.selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImg = document.getElementById('previewImg');
        previewImg.src = e.target.result;
        
        document.getElementById('imagePreview').style.display = 'block';
        document.getElementById('analysisControls').style.display = 'block';
        document.getElementById('uploadZone').style.display = 'none';
        
        // Show file info
        document.getElementById('previewInfo').textContent = 
            `${file.name} • ${(file.size / 1024).toFixed(1)} KB • ${file.type}`;
        
        // Re-init icons
        initLucideIcons();
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    state.selectedFile = null;
    state.analysisResult = null;
    
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('analysisControls').style.display = 'none';
    document.getElementById('uploadZone').style.display = 'block';
    document.getElementById('resultsPanel').style.display = 'none';
    document.getElementById('fileInput').value = '';
    
    initLucideIcons();
}

// ═══════════════════════════════════════════
// Analysis Pipeline
// ═══════════════════════════════════════════

async function runAnalysis() {
    if (!state.selectedFile || state.isAnalyzing) return;
    
    state.isAnalyzing = true;
    
    const enhance = document.getElementById('enhanceToggle').checked;
    const colormap = document.getElementById('colormapSelect').value;
    
    // Show results panel with loading
    const resultsPanel = document.getElementById('resultsPanel');
    const analysisLoading = document.getElementById('analysisLoading');
    const resultsContent = document.getElementById('resultsContent');
    
    resultsPanel.style.display = 'block';
    analysisLoading.style.display = 'block';
    resultsContent.style.display = 'none';
    
    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Animate pipeline steps
    initLucideIcons();
    await animatePipelineSteps();
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', state.selectedFile);
        
        // Make API request
        const response = await fetch(
            `${CONFIG.API_BASE}/api/analyze?enhance=${enhance}&colormap=${colormap}`,
            { method: 'POST', body: formData }
        );
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        state.analysisResult = result;
        
        // Complete all steps
        completeAllPipelineSteps();
        
        // Short delay then show results
        await sleep(500);
        
        analysisLoading.style.display = 'none';
        resultsContent.style.display = 'block';
        
        // Render all results
        renderResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
        analysisLoading.style.display = 'none';
    }
    
    state.isAnalyzing = false;
}

async function animatePipelineSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
    
    // Reset all steps
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'completed');
    });
    
    // Animate each step
    for (let i = 0; i < steps.length; i++) {
        const el = document.getElementById(steps[i]);
        el.classList.add('active');
        initLucideIcons();
        
        await sleep(CONFIG.ANIMATION_DELAY);
        
        el.classList.remove('active');
        el.classList.add('completed');
        
        initLucideIcons();
    }
}

function completeAllPipelineSteps() {
    ['step1', 'step2', 'step3', 'step4', 'step5'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active');
        el.classList.add('completed');
    });
}

// ═══════════════════════════════════════════
// Results Rendering
// ═══════════════════════════════════════════

function renderResults(result) {
    renderResultHeader(result);
    renderRiskCard(result);
    renderHeatmaps(result);
    renderFindings(result);
    renderCharts(result);
    renderClinicalReport(result);
    initLucideIcons();
}

function renderResultHeader(result) {
    const header = document.getElementById('resultHeader');
    header.querySelector('.result-id').textContent = result.analysis_id;
    header.querySelector('.result-time').textContent = 
        `Processed in ${result.processing_time_ms}ms`;
}

function renderRiskCard(result) {
    const assessment = result.predictions?.assessment || {};
    const riskCard = document.getElementById('riskCard');
    const riskScore = assessment.risk_score || 0;
    const statusCode = assessment.status_code || 'normal';
    
    // Update card class
    riskCard.className = `card risk-card ${statusCode}`;
    
    // Animate risk score
    animateNumber('riskScore', 0, riskScore, 1500);
    
    // Update status text
    document.getElementById('riskStatus').textContent = assessment.status || 'Analysis Complete';
    document.getElementById('riskRecommendation').textContent = assessment.recommendation || '';
    
    // Draw gauge
    drawRiskGauge(riskScore, statusCode);
    
    // Color the score
    const scoreEl = document.getElementById('riskScore');
    const colors = {
        critical: '#ef4444',
        abnormal: '#f59e0b',
        borderline: '#eab308',
        normal: '#22c55e'
    };
    scoreEl.style.color = colors[statusCode] || '#3b82f6';
}

function drawRiskGauge(value, statusCode) {
    const canvas = document.getElementById('riskGauge');
    const ctx = canvas.getContext('2d');
    
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height - 10;
    const radius = 80;
    
    ctx.clearRect(0, 0, width, height);
    
    // Background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 0, false);
    ctx.lineWidth = 12;
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Value arc
    const angle = Math.PI + (value / 100) * Math.PI;
    
    const gradientColors = {
        critical: ['#ef4444', '#f87171'],
        abnormal: ['#f59e0b', '#fb923c'],
        borderline: ['#eab308', '#f59e0b'],
        normal: ['#22c55e', '#3b82f6']
    };
    
    const colors = gradientColors[statusCode] || gradientColors.normal;
    const gradient = ctx.createLinearGradient(centerX - radius, centerY, centerX + radius, centerY);
    gradient.addColorStop(0, colors[0]);
    gradient.addColorStop(1, colors[1]);
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, angle, false);
    ctx.lineWidth = 12;
    ctx.strokeStyle = gradient;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Glow effect
    ctx.shadowColor = colors[0];
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, angle, false);
    ctx.lineWidth = 4;
    ctx.strokeStyle = colors[0];
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Tick marks
    for (let i = 0; i <= 10; i++) {
        const tickAngle = Math.PI + (i / 10) * Math.PI;
        const innerR = radius - 18;
        const outerR = radius - 22;
        
        ctx.beginPath();
        ctx.moveTo(
            centerX + Math.cos(tickAngle) * innerR,
            centerY + Math.sin(tickAngle) * innerR
        );
        ctx.lineTo(
            centerX + Math.cos(tickAngle) * outerR,
            centerY + Math.sin(tickAngle) * outerR
        );
        ctx.strokeStyle = '#d1d5db';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

function renderHeatmaps(result) {
    const heatmaps = result.heatmaps?.primary || {};
    
    // Store heatmap images
    state.heatmapImages = {
        overlay: heatmaps.overlay ? `data:image/png;base64,${heatmaps.overlay}` : '',
        heatmap: heatmaps.raw ? `data:image/png;base64,${heatmaps.raw}` : '',
        contours: heatmaps.contours ? `data:image/png;base64,${heatmaps.contours}` : '',
        original: document.getElementById('previewImg').src
    };
    
    // Show overlay by default
    const heatmapImg = document.getElementById('heatmapImage');
    heatmapImg.src = state.heatmapImages.overlay || state.heatmapImages.original;
    
    // Setup heatmap view buttons
    document.querySelectorAll('.hmap-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.hmap-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const view = btn.getAttribute('data-view');
            heatmapImg.src = state.heatmapImages[view] || state.heatmapImages.original;
        });
    });
    
    // Render attention regions
    renderAttentionRegions(heatmaps.attention_regions || []);
}

function renderAttentionRegions(regions) {
    const container = document.getElementById('attentionRegions');
    
    if (regions.length === 0) {
        container.innerHTML = '<p style="font-size:13px;color:#9ca3af;padding:8px;">No significant attention regions detected.</p>';
        return;
    }
    
    container.innerHTML = `
        <h4 style="font-size:13px;font-weight:600;margin-bottom:10px;color:#6b7280;">
            Attention Regions (${regions.length})
        </h4>
        ${regions.map(r => `
            <div class="region-item">
                <span class="region-location">${r.location}</span>
                <span class="region-strength ${r.significance}">${r.activation_strength}%</span>
            </div>
        `).join('')}
    `;
}

function renderFindings(result) {
    const findings = result.predictions?.findings || [];
    const container = document.getElementById('findingsList');
    
    const significantFindings = findings.filter(f => f.confidence >= 5);
    
    container.innerHTML = significantFindings.map((f, i) => {
        const color = getConfidenceColor(f.confidence);
        const severityIcons = {
            critical: '🔴',
            high: '🟠',
            moderate: '🟡',
            low: '🟢',
            minimal: '⚪'
        };
        
        return `
            <div class="finding-card" style="animation: fadeInUp 0.3s ease ${i * 0.05}s both;">
                <div class="finding-severity">${severityIcons[f.severity] || '⚪'}</div>
                <div class="finding-info">
                    <div class="finding-name">${f.condition}</div>
                    <div class="finding-desc">${f.description}</div>
                </div>
                <div class="finding-confidence">
                    <span class="confidence-value" style="color:${color}">${f.confidence}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:${f.confidence}%;background:${color};"></div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function renderCharts(result) {
    const findings = result.predictions?.findings || [];
    const quality = result.quality_assessment || {};
    
    // Destroy existing charts
    Object.values(state.charts).forEach(chart => chart.destroy());
    state.charts = {};
    
    // Confidence Bar Chart
    renderConfidenceChart(findings);
    
    // Quality Radar Chart
    renderQualityChart(quality);
}

function renderConfidenceChart(findings) {
    const ctx = document.getElementById('confidenceChart')?.getContext('2d');
    if (!ctx) return;
    
    const significant = findings.filter(f => f.confidence >= 5).slice(0, 10);
    
    const colors = significant.map(f => {
        if (f.confidence >= 65) return 'rgba(239, 68, 68, 0.75)';
        if (f.confidence >= 40) return 'rgba(245, 158, 11, 0.75)';
        if (f.confidence >= 20) return 'rgba(234, 179, 8, 0.75)';
        return 'rgba(34, 197, 94, 0.75)';
    });
    
    const borderColors = significant.map(f => {
        if (f.confidence >= 65) return '#ef4444';
        if (f.confidence >= 40) return '#f59e0b';
        if (f.confidence >= 20) return '#eab308';
        return '#22c55e';
    });
    
    state.charts.confidence = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: significant.map(f => f.condition),
            datasets: [{
                label: 'Confidence (%)',
                data: significant.map(f => f.confidence),
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.7
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1f2937',
                    borderColor: '#e5e7eb',
                    borderWidth: 1,
                    titleColor: '#f9fafb',
                    bodyColor: '#d1d5db',
                    titleFont: { family: 'Inter', weight: '600' },
                    padding: 12,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    max: 100,
                    grid: { color: '#f3f4f6' },
                    ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 12 } }
                }
            }
        }
    });
    
    // Set chart canvas height
    document.getElementById('confidenceChart').parentElement.style.height = 
        `${Math.max(300, significant.length * 36)}px`;
}

function renderQualityChart(quality) {
    const ctx = document.getElementById('qualityChart')?.getContext('2d');
    if (!ctx) return;
    
    const brightness = Math.min(quality.brightness || 0, 100);
    const contrast = Math.min(quality.contrast || 0, 100);
    const sharpness = Math.min((quality.sharpness || 0) / 10, 100);
    const noise = Math.max(0, 100 - (quality.noise_level || 0) * 3);
    const overall = quality.quality_score || 0;
    
    state.charts.quality = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Brightness', 'Contrast', 'Sharpness', 'Noise Level', 'Overall'],
            datasets: [{
                label: 'Quality Metrics',
                data: [brightness, contrast, sharpness, noise, overall],
                backgroundColor: 'rgba(59, 130, 246, 0.08)',
                borderColor: '#3b82f6',
                borderWidth: 2,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1f2937',
                    borderColor: '#e5e7eb',
                    borderWidth: 1,
                    titleColor: '#f9fafb',
                    bodyColor: '#d1d5db',
                    padding: 12,
                    cornerRadius: 8
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: '#f3f4f6' },
                    angleLines: { color: '#e5e7eb' },
                    pointLabels: {
                        color: '#6b7280',
                        font: { family: 'Inter', size: 12 }
                    },
                    ticks: {
                        display: false,
                        stepSize: 25
                    }
                }
            }
        }
    });
}

function renderClinicalReport(result) {
    const report = result.clinical_report || {};
    const container = document.getElementById('clinicalReport');
    
    const findings = report.findings || [];
    const recommendations = report.recommendations || [];
    const differentials = report.differential_diagnosis || [];
    
    container.innerHTML = `
        <!-- Report Header -->
        <div class="report-section">
            <div class="report-title">Report Information</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
                <div><span style="color:var(--text-muted)">Report ID:</span> <span style="color:var(--accent-cyan);font-family:var(--font-mono);">${report.report_id || 'N/A'}</span></div>
                <div><span style="color:var(--text-muted)">Date:</span> ${report.generated_at_formatted || 'N/A'}</div>
                <div><span style="color:var(--text-muted)">Priority:</span> ${report.header?.priority || 'N/A'}</div>
                <div><span style="color:var(--text-muted)">System:</span> ${report.header?.system_version || 'N/A'}</div>
            </div>
        </div>

        <!-- Image Quality -->
        <div class="report-section">
            <div class="report-title">Image Quality Assessment</div>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <span style="font-size:28px;font-weight:800;color:${report.image_quality?.diagnostic_quality ? 'var(--accent-green)' : 'var(--accent-red)'}">
                    ${report.image_quality?.overall_score || 0}/100
                </span>
                <span style="font-size:14px;color:var(--text-secondary)">${report.image_quality?.grade || 'N/A'}</span>
                <span style="font-size:12px;padding:2px 8px;border-radius:8px;background:${report.image_quality?.diagnostic_quality ? 'rgba(67,233,123,0.1)' : 'rgba(255,71,87,0.1)'};color:${report.image_quality?.diagnostic_quality ? 'var(--accent-green)' : 'var(--accent-red)'}">
                    ${report.image_quality?.diagnostic_quality ? '✓ Diagnostic Quality' : '✗ Below Diagnostic Quality'}
                </span>
            </div>
        </div>

        <!-- Impression -->
        <div class="report-section">
            <div class="report-title">Clinical Impression</div>
            <p class="report-text">${report.impression || 'No impression available.'}</p>
        </div>

        <!-- Key Findings -->
        <div class="report-section">
            <div class="report-title">Key Findings (${findings.length})</div>
            ${findings.slice(0, 8).map(f => `
                <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;margin-bottom:4px;background:rgba(0,212,255,0.02);border-radius:8px;border:1px solid var(--border-subtle);">
                    <div>
                        <span>${f.severity_icon || '⚪'}</span>
                        <span style="font-weight:600;font-size:14px;margin-left:4px;">${f.condition}</span>
                        <span style="font-size:11px;color:var(--text-muted);margin-left:8px;">${f.icd10_code || ''}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-family:var(--font-mono);font-weight:600;font-size:14px;color:${getConfidenceColor(f.confidence_percent)}">${f.confidence_percent}%</span>
                    </div>
                </div>
            `).join('')}
        </div>

        <!-- Attention Analysis -->
        <div class="report-section">
            <div class="report-title">AI Attention Analysis</div>
            <p class="report-text">${report.attention_analysis?.interpretation || 'No attention analysis available.'}</p>
        </div>

        <!-- Differential Diagnosis -->
        ${differentials.length > 0 ? `
        <div class="report-section">
            <div class="report-title">Differential Diagnosis</div>
            ${differentials.slice(0, 4).map(d => `
                <div style="margin-bottom:12px;padding:12px;background:rgba(0,212,255,0.02);border-radius:8px;border:1px solid var(--border-subtle);">
                    <div style="font-weight:600;font-size:14px;margin-bottom:6px;">${d.primary} (${d.confidence}%)</div>
                    <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;">Differentials: ${d.differentials?.join(', ') || 'N/A'}</div>
                    <div style="font-size:12px;color:var(--text-muted);">Suggested workup: ${d.suggested_workup?.join(', ') || 'N/A'}</div>
                </div>
            `).join('')}
        </div>
        ` : ''}

        <!-- Recommendations -->
        <div class="report-section">
            <div class="report-title">Recommendations</div>
            <ul class="report-list">
                ${recommendations.map(r => `<li>${r}</li>`).join('')}
            </ul>
        </div>

        <!-- Technical Details -->
        <div class="report-section">
            <div class="report-title">Technical Details</div>
            <div style="font-size:12px;color:var(--text-muted);font-family:var(--font-mono);line-height:1.8;">
                <div>Algorithm: ${report.technical_details?.algorithm || 'N/A'}</div>
                <div>Training Data: ${report.technical_details?.training_data || 'N/A'}</div>
                <div>Explainability: ${report.technical_details?.explainability || 'N/A'}</div>
                <div>Target Layer: ${report.technical_details?.target_layer || 'N/A'}</div>
            </div>
        </div>

        <!-- References -->
        ${report.references ? `
        <div class="report-section">
            <div class="report-title">References</div>
            <ul class="report-list">
                ${report.references.map(r => `
                    <li style="font-size:12px;">${r.authors} "${r.title}" ${r.journal}, ${r.year}</li>
                `).join('')}
            </ul>
        </div>
        ` : ''}

        <!-- Disclaimer -->
        <div class="report-disclaimer">
            ${report.disclaimer || 'This analysis is for research/educational purposes only.'}
        </div>
    `;
}

// ═══════════════════════════════════════════
// Tabs
// ═══════════════════════════════════════════

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Update button states
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            document.getElementById(`tab-${tabId}`).classList.add('active');
            
            initLucideIcons();
        });
    });
}

// ═══════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function animateNumber(elementId, start, end, duration) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;
        
        element.textContent = Math.round(current);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

function getConfidenceColor(confidence) {
    if (confidence >= 65) return '#ff4757';
    if (confidence >= 40) return '#ffa502';
    if (confidence >= 20) return '#ffd32a';
    return '#43e97b';
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 24px;
        padding: 14px 24px;
        background: ${type === 'error' ? 'rgba(255, 71, 87, 0.15)' : 'rgba(0, 212, 255, 0.1)'};
        border: 1px solid ${type === 'error' ? 'rgba(255, 71, 87, 0.4)' : 'rgba(0, 212, 255, 0.3)'};
        color: ${type === 'error' ? '#ff6b81' : '#00d4ff'};
        border-radius: 12px;
        font-size: 14px;
        font-family: Inter, sans-serif;
        z-index: 10000;
        backdrop-filter: blur(20px);
        animation: fadeInUp 0.3s ease;
        max-width: 400px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-10px)';
        notification.style.transition = 'all 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}
