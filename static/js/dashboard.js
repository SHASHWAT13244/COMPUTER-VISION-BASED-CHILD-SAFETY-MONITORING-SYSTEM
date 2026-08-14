// static/js/dashboard.js
const socket = io();

let activityChart = null;

// DOM Elements
const dashStatus = document.getElementById('dashStatus');
const dashActivity = document.getElementById('dashActivity');
const dashSafety = document.getElementById('dashSafety');
const dashAlerts = document.getElementById('dashAlerts');
const alertTimeline = document.getElementById('alertTimeline');
const captureGallery = document.getElementById('captureGallery');

// Socket events
socket.on('status_update', (data) => {
    updateDashboard(data);
});

socket.on('alert', (data) => {
    addAlertToTimeline(data);
    updateAlertCount();
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
    if (data.monitoring !== undefined) {
        dashStatus.textContent = data.monitoring ? 'Active' : 'Inactive';
        dashStatus.className = data.monitoring ? 'text-success' : 'text-danger';
    }
    
    if (data.activity !== undefined) {
        dashActivity.textContent = data.activity || 'None';
    }
    
    if (data.safe !== undefined) {
        dashSafety.textContent = data.safe ? '✅ Safe' : '⚠️ Unsafe';
        dashSafety.className = data.safe ? 'text-success' : 'text-danger';
    }
    
    if (data.alert_count !== undefined) {
        dashAlerts.textContent = data.alert_count;
    }
}

function addAlertToTimeline(data) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert-item ${data.severity || 'medium'} mb-2`;
    alertDiv.innerHTML = `
        <div class="d-flex justify-content-between">
            <span><i class="fas fa-${data.severity === 'high' ? 'exclamation-triangle' : 'exclamation-circle'}"></i> ${data.message}</span>
            <span class="badge bg-${data.severity === 'high' ? 'danger' : 'warning'}">${data.severity}</span>
        </div>
        <div class="timestamp">${data.timestamp}</div>
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
                // Update chart with alert statistics
                updateAlertStats(data.alerts);
            }
        })
        .catch(error => console.error('Error fetching alerts:', error));
}

function updateAlertStats(alerts) {
    const severityCount = {};
    const activityCount = {};
    
    alerts.forEach(alert => {
        severityCount[alert.severity] = (severityCount[alert.severity] || 0) + 1;
        if (alert.activity) {
            activityCount[alert.activity] = (activityCount[alert.activity] || 0) + 1;
        }
    });
    
    // Update charts if they exist
    // This would require additional chart instances
}

// Refresh interval
setInterval(fetchDashboardData, 5000);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initActivityChart();
    fetchDashboardData();
    updateAlertCount();
});