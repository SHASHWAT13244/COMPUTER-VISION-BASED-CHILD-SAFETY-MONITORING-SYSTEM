// static/js/dashboard.js
const socket = io();

let activityChart = null;
let imageAnalysisHistory = [];
let imageAnalysisChart = null;

// DOM Elements
const dashStatus = document.getElementById('dashStatus');
const dashActivity = document.getElementById('dashActivity');
const dashSafety = document.getElementById('dashSafety');
const dashAlerts = document.getElementById('dashAlerts');
const alertTimeline = document.getElementById('alertTimeline');
const captureGallery = document.getElementById('captureGallery');

// Socket events
socket.on('status_update', (data) => {
    console.log('📊 Dashboard status update:', data);
    updateDashboard(data);
    
    // Update dashboard for image analysis
    if (data.source === 'image_upload' && data.activity && data.activity !== 'None') {
        addImageAnalysisToHistory(data);
        updateImageAnalysisChart(data);
        updateActivityChart(data);
    }
});

socket.on('alert', (data) => {
    console.log('🚨 Dashboard alert received:', data);
    addAlertToTimeline(data);
    updateAlertCount();
    
    // If alert is from image upload, highlight it
    if (data.source === 'image_upload') {
        highlightImageAlert(data);
    }
});

// Initialize chart
function initActivityChart() {
    const ctx = document.getElementById('activityChartCanvas').getContext('2d');
    activityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Walking', 'Running', 'Sitting', 'Falling', 'Climbing'],
            datasets: [{
                data: [0, 0, 0, 0, 0],
                backgroundColor: ['#4CAF50', '#FF9800', '#2196F3', '#F44336', '#9C27B0']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function updateDashboard(data) {
    // Update status
    if (data.monitoring !== undefined) {
        dashStatus.textContent = data.monitoring ? 'Active' : 'Inactive';
        dashStatus.className = data.monitoring ? 'text-success' : 'text-danger';
    }
    
    // Update activity - FIXED
    if (data.activity !== undefined) {
        const activityDisplay = data.activity || 'None';
        dashActivity.textContent = activityDisplay;
        
        // Highlight if from image analysis
        if (data.source === 'image_upload' && data.activity !== 'None') {
            dashActivity.className = 'text-warning';
            dashActivity.title = 'From Image Analysis';
        } else {
            dashActivity.className = '';
        }
    }
    
    // Update safety
    if (data.safe !== undefined) {
        dashSafety.textContent = data.safe ? '✅ Safe' : '⚠️ Unsafe';
        dashSafety.className = data.safe ? 'text-success' : 'text-danger';
    }
    
    // Update alert count
    if (data.alert_count !== undefined) {
        dashAlerts.textContent = data.alert_count;
    }
}

function updateActivityChart(data) {
    if (!activityChart) return;
    
    const labels = ['Walking', 'Running', 'Sitting', 'Falling', 'Climbing'];
    const activityMap = {
        'walking': 0,
        'running': 1,
        'sitting': 2,
        'falling': 3,
        'climbing': 4
    };
    
    const index = activityMap[data.activity?.toLowerCase()];
    if (index !== undefined) {
        const currentData = activityChart.data.datasets[0].data;
        // Add weight to show image analysis impact
        // We'll update this when we fetch full stats
        currentData[index] = currentData[index] + 0.5;
        activityChart.update();
    }
}

function addAlertToTimeline(data) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert-item ${data.severity || 'medium'} mb-2`;
    
    const sourceText = data.source === 'image_upload' ? '📷 Image' : '🎥 Live';
    
    alertDiv.innerHTML = `
        <div class="d-flex justify-content-between">
            <span><i class="fas fa-${data.severity === 'high' ? 'exclamation-triangle' : 'exclamation-circle'}"></i> ${data.message}</span>
            <span class="badge bg-${data.severity === 'high' ? 'danger' : 'warning'}">${data.severity}</span>
        </div>
        <div class="timestamp">
            ${data.timestamp || new Date().toLocaleString()}
            <span class="badge bg-secondary ms-1">${sourceText}</span>
        </div>
    `;
    
    alertTimeline.prepend(alertDiv);
    
    // Keep only last 20 alerts
    while (alertTimeline.children.length > 20) {
        alertTimeline.removeChild(alertTimeline.lastChild);
    }
}

function updateAlertCount() {
    const count = alertTimeline.children.length;
    // Update badge if exists
    const badge = document.querySelector('.badge.bg-danger');
    if (badge) {
        badge.textContent = count;
        badge.style.display = count > 0 ? 'inline' : 'none';
    }
}

function addImageAnalysisToHistory(data) {
    imageAnalysisHistory.push({
        timestamp: data.timestamp || new Date().toISOString(),
        activity: data.activity,
        confidence: data.confidence,
        safe: data.safe
    });
    
    // Keep only last 50 entries
    if (imageAnalysisHistory.length > 50) {
        imageAnalysisHistory.shift();
    }
}

function updateImageAnalysisChart(data) {
    // Placeholder for future chart implementation
    console.log('📊 Image analysis chart update:', data);
}

function highlightImageAlert(data) {
    const alertItems = alertTimeline.querySelectorAll('.alert-item');
    if (alertItems.length > 0) {
        const lastAlert = alertItems[0];
        lastAlert.style.borderLeftColor = '#9C27B0';
        lastAlert.style.background = '#f3e5f5';
        
        // Add badge
        const timestamp = lastAlert.querySelector('.timestamp');
        if (timestamp) {
            timestamp.innerHTML += ' <span class="badge bg-purple">📷 Image</span>';
        }
        
        // Reset after 5 seconds
        setTimeout(() => {
            lastAlert.style.borderLeftColor = '';
            lastAlert.style.background = '';
        }, 5000);
    }
}

// Fetch dashboard data
function fetchDashboardData() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateDashboard(data);
            if (data.alerts) {
                data.alerts.forEach(alert => addAlertToTimeline(alert));
            }
        })
        .catch(error => console.error('Error fetching dashboard data:', error));
    
    fetch('/api/alerts')
        .then(response => response.json())
        .then(data => {
            if (data.alerts && data.alerts.length > 0) {
                updateAlertStats(data.alerts);
            }
        })
        .catch(error => console.error('Error fetching alerts:', error));
}

function updateAlertStats(alerts) {
    // Update activity chart with alert statistics
    if (activityChart) {
        const counts = {
            'walking': 0,
            'running': 0,
            'sitting': 0,
            'falling': 0,
            'climbing': 0
        };
        
        alerts.forEach(alert => {
            if (alert.activity && counts[alert.activity.toLowerCase()] !== undefined) {
                counts[alert.activity.toLowerCase()]++;
            }
        });
        
        activityChart.data.datasets[0].data = Object.values(counts);
        activityChart.update();
    }
}

// Refresh interval
setInterval(fetchDashboardData, 5000);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('📊 Initializing Dashboard');
    initActivityChart();
    fetchDashboardData();
    updateAlertCount();
});
