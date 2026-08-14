// static/js/script.js
const socket = io();

let isMonitoring = false;

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const captureBtn = document.getElementById('captureBtn');
const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const uploadResult = document.getElementById('uploadResult');
const alertContainer = document.getElementById('alertContainer');
const alertBadge = document.getElementById('alertBadge');

// Status elements
const currentActivity = document.getElementById('currentActivity');
const currentConfidence = document.getElementById('currentConfidence');
const safetyStatus = document.getElementById('safetyStatus');
const alertCount = document.getElementById('alertCount');
const fpsDisplay = document.getElementById('fpsDisplay');
const statusBadge = document.getElementById('statusBadge');

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
});

socket.on('status_update', (data) => {
    updateStatus(data);
});

socket.on('alert', (data) => {
    addAlert(data);
    updateAlertBadge();
});

// Button event handlers
startBtn.addEventListener('click', () => {
    fetch('/api/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started' || data.status === 'already_running') {
                isMonitoring = true;
                updateStatusUI(true);
            }
        })
        .catch(error => console.error('Error starting monitoring:', error));
});

stopBtn.addEventListener('click', () => {
    fetch('/api/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                isMonitoring = false;
                updateStatusUI(false);
            }
        })
        .catch(error => console.error('Error stopping monitoring:', error));
});

resetBtn.addEventListener('click', () => {
    fetch('/api/reset', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'reset') {
                location.reload();
            }
        })
        .catch(error => console.error('Error resetting system:', error));
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

uploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.result) {
            displayUploadResult(data);
        } else {
            showNotification('Error processing image', 'danger');
        }
    })
    .catch(error => console.error('Error uploading image:', error));
});

// Status update functions
function updateStatus(data) {
    if (data.activity !== undefined) {
        currentActivity.textContent = data.activity || 'None';
    }
    if (data.confidence !== undefined) {
        currentConfidence.textContent = Math.round(data.confidence * 100) + '%';
    }
    if (data.safe !== undefined) {
        safetyStatus.textContent = data.safe ? '✅ Safe' : '⚠️ Unsafe';
        safetyStatus.className = 'value ' + (data.safe ? 'safe' : 'danger');
    }
    if (data.alert_count !== undefined) {
        alertCount.textContent = data.alert_count;
    }
}

function updateStatusUI(active) {
    if (active) {
        statusBadge.className = 'status-badge active';
        statusBadge.innerHTML = '<i class="fas fa-circle"></i> Active';
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } else {
        statusBadge.className = 'status-badge inactive';
        statusBadge.innerHTML = '<i class="fas fa-circle"></i> Inactive';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

function addAlert(data) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert-item ' + (data.severity || 'medium');
    alertDiv.innerHTML = `
        <div class="d-flex justify-content-between">
            <span><strong>${data.message}</strong></span>
            <span class="badge bg-${data.severity === 'high' ? 'danger' : data.severity === 'medium' ? 'warning' : 'info'}">${data.severity || 'unknown'}</span>
        </div>
        <div class="timestamp">${data.timestamp} - Activity: ${data.activity || 'Unknown'}</div>
    `;
    
    alertContainer.prepend(alertDiv);
    
    // Keep only last 20 alerts
    while (alertContainer.children.length > 20) {
        alertContainer.removeChild(alertContainer.lastChild);
    }
}

function updateAlertBadge() {
    const count = alertContainer.children.length;
    alertBadge.textContent = count;
    alertBadge.style.display = count > 0 ? 'inline' : 'none';
}

function displayUploadResult(data) {
    const result = data.result;
    uploadResult.innerHTML = `
        <div class="alert alert-${result.safe ? 'success' : 'danger'}">
            <h6>Analysis Result:</h6>
            <p><strong>Activity:</strong> ${result.activity || 'None'}</p>
            <p><strong>Confidence:</strong> ${Math.round(result.confidence * 100)}%</p>
            <p><strong>Safety:</strong> ${result.safe ? '✅ Safe' : '⚠️ Unsafe'}</p>
            ${result.message ? `<p><strong>Message:</strong> ${result.message}</p>` : ''}
            ${result.processed_image ? `<img src="data:image/jpeg;base64,${result.processed_image}" alt="Processed Image">` : ''}
        </div>
    `;
}

function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 80px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}

// Initial status update
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

// Update FPS periodically
setInterval(() => {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data.fps !== undefined) {
                fpsDisplay.textContent = data.fps;
            }
        })
        .catch(error => console.error('Error fetching FPS:', error));
}, 2000);

// Initialize
fetchInitialStatus();
updateStatusUI(false);