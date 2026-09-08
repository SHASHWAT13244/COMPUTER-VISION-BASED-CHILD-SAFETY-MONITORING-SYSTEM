// static/js/script.js
const socket = io();

let isMonitoring = false;
let confidenceColor = 'low';

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const captureBtn = document.getElementById('captureBtn');
const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const uploadResult = document.getElementById('uploadResult');
const uploadBtn = document.getElementById('uploadBtn');
const alertContainer = document.getElementById('alertContainer');
const alertBadge = document.getElementById('alertBadge');
const alertCountBadge = document.getElementById('alertCountBadge');

// Status elements
const currentActivity = document.getElementById('currentActivity');
const currentConfidence = document.getElementById('currentConfidence');
const safetyStatus = document.getElementById('safetyStatus');
const alertCount = document.getElementById('alertCount');
const fpsDisplay = document.getElementById('fpsDisplay');
const statusBadge = document.getElementById('statusBadge');
const overlayBadge = document.getElementById('overlayBadge');

// ==================== SOCKET EVENT HANDLERS ====================

socket.on('connect', () => {
    console.log('✅ Connected to server');
});

socket.on('disconnect', () => {
    console.log('❌ Disconnected from server');
});

socket.on('status_update', (data) => {
    updateStatus(data);
});

socket.on('alert', (data) => {
    addAlert(data);
    updateAlertBadge();
});

// ==================== BUTTON EVENT HANDLERS ====================

startBtn.addEventListener('click', () => {
    startBtn.disabled = true;
    startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
    
    fetch('/api/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started' || data.status === 'already_running') {
                isMonitoring = true;
                updateStatusUI(true);
                showNotification('Monitoring started successfully', 'success');
            } else {
                showNotification('Failed to start monitoring', 'danger');
                startBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-play"></i> Start';
            }
        })
        .catch(error => {
            console.error('Error starting monitoring:', error);
            showNotification('Error starting monitoring', 'danger');
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-play"></i> Start';
        });
});

stopBtn.addEventListener('click', () => {
    stopBtn.disabled = true;
    stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';
    
    fetch('/api/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                isMonitoring = false;
                updateStatusUI(false);
                showNotification('Monitoring stopped', 'info');
            }
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
        })
        .catch(error => {
            console.error('Error stopping monitoring:', error);
            showNotification('Error stopping monitoring', 'danger');
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
        });
});

resetBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to reset the system?')) {
        fetch('/api/reset', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'reset') {
                    location.reload();
                }
            })
            .catch(error => console.error('Error resetting system:', error));
    }
});

captureBtn.addEventListener('click', () => {
    fetch('/api/capture', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.filename) {
                showNotification('Image captured successfully!', 'success');
            }
        })
        .catch(error => console.error('Error capturing image:', error));
});

// ==================== IMAGE UPLOAD HANDLER ====================

uploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const file = imageInput.files[0];
    if (!file) {
        showNotification('Please select an image first', 'warning');
        return;
    }
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file', 'warning');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('Image size should be less than 10MB', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    // Show loading state
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    uploadResult.innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2 text-muted">Processing image...</p>
            <small class="text-muted">This may take a few seconds</small>
        </div>
    `;
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Analyze Image';
        
        if (data.error) {
            showNotification(data.error, 'danger');
            uploadResult.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
            return;
        }
        
        if (data.result) {
            displayUploadResult(data);
            showNotification('Analysis complete!', 'success');
        } else {
            uploadResult.innerHTML = '<div class="alert alert-warning">No results returned. Please try again.</div>';
        }
    })
    .catch(error => {
        console.error('Error uploading image:', error);
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Analyze Image';
        showNotification('Error uploading image', 'danger');
        uploadResult.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> Failed to process image. Please try again.
                <br><small>${error.message || 'Unknown error'}</small>
            </div>
        `;
    });
});

// ==================== DISPLAY FUNCTIONS ====================

function displayUploadResult(data) {
    const result = data.result;
    const confidencePercent = Math.round((result.confidence || 0) * 100);
    
    // Determine confidence color
    let confidenceClass = 'confidence-low';
    if (confidencePercent > 70) confidenceClass = 'confidence-high';
    else if (confidencePercent > 40) confidenceClass = 'confidence-medium';
    
    let html = `<div class="alert alert-${result.safe ? 'success' : 'danger'}">`;
    
    // Header
    html += `<h6 class="mb-2"><i class="fas fa-${result.safe ? 'check-circle' : 'exclamation-triangle'}"></i> Analysis Result</h6>`;
    
    // Activity and Confidence
    html += `
        <div class="row g-2 mb-2">
            <div class="col-6">
                <small class="text-muted">Activity</small>
                <div class="fw-bold">${result.activity || 'None'}</div>
            </div>
            <div class="col-6">
                <small class="text-muted">Confidence</small>
                <div class="fw-bold ${confidenceClass}">${confidencePercent}%</div>
            </div>
        </div>
    `;
    
    // Safety status
    html += `
        <div class="row g-2 mb-2">
            <div class="col-6">
                <small class="text-muted">Safety</small>
                <div class="fw-bold ${result.safe ? 'text-success' : 'text-danger'}">
                    ${result.safe ? '✅ Safe' : '⚠️ Unsafe'}
                </div>
            </div>
            <div class="col-6">
                <small class="text-muted">Pose Detected</small>
                <div class="fw-bold ${result.pose_detected ? 'text-success' : 'text-warning'}">
                    ${result.pose_detected ? '✅ Yes' : '⚠️ No'}
                </div>
            </div>
        </div>
    `;
    
    // Message
    if (result.message && result.message !== 'All safe') {
        html += `<div class="mb-2"><small class="text-muted">Message</small><div>${result.message}</div></div>`;
    }
    
    // Activity Probabilities
    if (data.probabilities && Object.keys(data.probabilities).length > 0) {
        html += `<hr><div class="mt-2"><strong>Activity Probabilities:</strong>`;
        html += `<div class="mt-2">`;
        
        // Sort probabilities by value (descending)
        const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
        
        for (const [activity, prob] of sorted) {
            const probPercent = Math.round(prob * 100);
            let barColor = '#6c757d';
            let textColor = '#fff';
            
            if (probPercent > 70) {
                barColor = '#28a745';
                textColor = '#fff';
            } else if (probPercent > 40) {
                barColor = '#ffc107';
                textColor = '#212529';
            } else if (probPercent > 20) {
                barColor = '#fd7e14';
                textColor = '#fff';
            }
            
            // Highlight the predicted activity
            const isPredicted = result.activity && activity.toLowerCase() === result.activity.toLowerCase();
            const borderStyle = isPredicted ? 'border: 2px solid #28a745;' : '';
            const bgStyle = isPredicted ? 'background: #f0fff4;' : '';
            
            html += `
                <div class="probability-bar d-flex align-items-center mb-1" style="${bgStyle} ${borderStyle} border-radius: 4px; padding: 2px 4px;">
                    <span class="label" style="width: 80px; font-size: 12px; font-weight: ${isPredicted ? '700' : '400'};">
                        ${isPredicted ? '⭐ ' : ''}${activity}
                    </span>
                    <div class="progress flex-grow-1" style="height: 22px; border-radius: 4px;">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${probPercent}%; background-color: ${barColor}; color: ${textColor}; font-size: 11px; line-height: 22px; font-weight: 600;"
                             aria-valuenow="${probPercent}" aria-valuemin="0" aria-valuemax="100">
                            ${probPercent}%
                        </div>
                    </div>
                </div>
            `;
        }
        html += `</div></div>`;
    }
    
    // Processed Image
    if (result.processed_image) {
        html += `
            <div class="mt-2">
                <small class="text-muted">Processed Image:</small>
                <img src="data:image/jpeg;base64,${result.processed_image}" 
                     alt="Processed Image" 
                     style="max-width: 100%; border-radius: 8px; margin-top: 5px; border: 1px solid #ddd;">
            </div>
        `;
    }
    
    // Detections info
    if (result.detections && result.detections.length > 0) {
        html += `
            <div class="mt-2">
                <small class="text-muted">Detections: ${result.detections.length}</small>
            </div>
        `;
    }
    
    html += `</div>`;
    uploadResult.innerHTML = html;
}

// ==================== STATUS UPDATE FUNCTIONS ====================

function updateStatus(data) {
    if (data.activity !== undefined) {
        currentActivity.textContent = data.activity || 'None';
    }
    
    if (data.confidence !== undefined) {
        const confidencePercent = Math.round(data.confidence * 100);
        currentConfidence.textContent = confidencePercent + '%';
        
        // Update color based on confidence
        currentConfidence.className = 'value';
        if (confidencePercent > 70) {
            currentConfidence.classList.add('confidence-high');
        } else if (confidencePercent > 40) {
            currentConfidence.classList.add('confidence-medium');
        } else {
            currentConfidence.classList.add('confidence-low');
        }
    }
    
    if (data.safe !== undefined) {
        safetyStatus.textContent = data.safe ? '✅ Safe' : '⚠️ Unsafe';
        safetyStatus.className = 'value ' + (data.safe ? 'safe' : 'danger');
    }
    
    if (data.alert_count !== undefined) {
        alertCount.textContent = data.alert_count;
        if (alertCountBadge) alertCountBadge.textContent = data.alert_count;
    }
    
    if (data.fps !== undefined) {
        fpsDisplay.textContent = data.fps.toFixed(1);
    }
}

function updateStatusUI(active) {
    if (active) {
        statusBadge.className = 'status-badge active';
        statusBadge.innerHTML = '<i class="fas fa-circle"></i> Active';
        overlayBadge.className = 'status-badge active';
        overlayBadge.innerHTML = '<i class="fas fa-circle"></i> Active';
        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fas fa-play"></i> Start';
        stopBtn.disabled = false;
        stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
    } else {
        statusBadge.className = 'status-badge inactive';
        statusBadge.innerHTML = '<i class="fas fa-circle"></i> Inactive';
        overlayBadge.className = 'status-badge inactive';
        overlayBadge.innerHTML = '<i class="fas fa-circle"></i> Inactive';
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-play"></i> Start';
        stopBtn.disabled = true;
        stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
    }
}

function addAlert(data) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert-item ' + (data.severity || 'medium');
    
    const confidenceText = data.confidence ? ` (${Math.round(data.confidence * 100)}%)` : '';
    
    alertDiv.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <span><strong>${data.message}</strong></span>
            <span class="badge bg-${data.severity === 'high' ? 'danger' : data.severity === 'medium' ? 'warning' : 'info'}">${data.severity || 'unknown'}</span>
        </div>
        <div class="timestamp">${data.timestamp || new Date().toLocaleString()} - ${data.activity || 'Unknown'}${confidenceText}</div>
    `;
    
    alertContainer.prepend(alertDiv);
    
    // Remove "No alerts" message
    const noAlertsMsg = alertContainer.querySelector('.text-muted');
    if (noAlertsMsg) noAlertsMsg.remove();
    
    // Keep only last 20 alerts
    while (alertContainer.children.length > 20) {
        alertContainer.removeChild(alertContainer.lastChild);
    }
    
    updateAlertBadge();
}

function updateAlertBadge() {
    const count = alertContainer.querySelectorAll('.alert-item').length;
    if (alertBadge) {
        alertBadge.textContent = count;
        alertBadge.style.display = count > 0 ? 'inline' : 'none';
    }
    if (alertCountBadge) {
        alertCountBadge.textContent = count;
    }
}

// ==================== NOTIFICATION FUNCTION ====================

function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    const bgColor = {
        'success': 'bg-success',
        'danger': 'bg-danger',
        'warning': 'bg-warning',
        'info': 'bg-info'
    }[type] || 'bg-info';
    
    const textColor = (type === 'warning') ? 'text-dark' : 'text-white';
    
    alertDiv.className = `alert ${bgColor} ${textColor} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 80px; right: 20px; z-index: 9999; min-width: 300px; max-width: 500px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);';
    alertDiv.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            ${message}
        </div>
        <button type="button" class="btn-close ${textColor}" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    document.body.appendChild(alertDiv);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.classList.add('fade');
            setTimeout(() => {
                if (alertDiv.parentNode) alertDiv.remove();
            }, 300);
        }
    }, 5000);
}

// ==================== PERIODIC STATUS UPDATE ====================

function fetchInitialStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateStatus(data);
            if (data.monitoring_active) {
                isMonitoring = true;
                updateStatusUI(true);
            }
            if (data.alerts) {
                data.alerts.forEach(alert => addAlert(alert));
                updateAlertBadge();
            }
        })
        .catch(error => console.error('Error fetching status:', error));
}

// Refresh status every 2 seconds
setInterval(() => {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateStatus(data);
        })
        .catch(error => console.error('Error fetching status:', error));
}, 2000);

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', () => {
    fetchInitialStatus();
    updateStatusUI(false);
    
    // File input change handler - show filename
    imageInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const label = this.nextElementSibling;
            if (label && label.classList.contains('form-text')) {
                label.textContent = `Selected: ${this.files[0].name} (${(this.files[0].size / 1024).toFixed(1)} KB)`;
            }
        }
    });
});

console.log('✅ Child Safety Monitoring System initialized');
