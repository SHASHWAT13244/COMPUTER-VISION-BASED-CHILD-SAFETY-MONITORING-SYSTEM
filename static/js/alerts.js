// static/js/alerts.js
const socket = io();

let statsChart = null;
let severityChart = null;

// DOM Elements
const alertTableBody = document.getElementById('alertTableBody');
const alertSummary = document.getElementById('alertSummary');
const refreshBtn = document.getElementById('refreshAlerts');
const clearBtn = document.getElementById('clearAlerts');
const exportBtn = document.getElementById('exportAlerts');

// Socket events
socket.on('alert', (data) => {
    addAlertToTable(data);
    updateAlertSummary();
    updateCharts();
});

// Initialize charts
function initCharts() {
    // Stats chart
    const statsCtx = document.getElementById('alertStatsChart').getContext('2d');
    statsChart = new Chart(statsCtx, {
        type: 'bar',
        data: {
            labels: ['Walking', 'Running', 'Sitting', 'Falling', 'Climbing'],
            datasets: [{
                label: 'Alert Count',
                data: [0, 0, 0, 0, 0],
                backgroundColor: ['#4CAF50', '#FF9800', '#2196F3', '#F44336', '#9C27B0']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Severity chart
    const severityCtx = document.getElementById('severityChart').getContext('2d');
    severityChart = new Chart(severityCtx, {
        type: 'pie',
        data: {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#F44336', '#FF9800', '#4CAF50']
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

function addAlertToTable(data) {
    const row = document.createElement('tr');
    const count = alertTableBody.children.length + 1;
    
    const severityClass = data.severity === 'high' ? 'danger' : 
                         data.severity === 'medium' ? 'warning' : 'info';
    
    row.innerHTML = `
        <td>${count}</td>
        <td>${data.timestamp}</td>
        <td><span class="badge bg-${severityClass}">${data.severity || 'unknown'}</span></td>
        <td>${data.activity || 'Unknown'}</td>
        <td>${data.message}</td>
        <td><span class="badge bg-secondary">New</span></td>
    `;
    
    alertTableBody.prepend(row);
    
    // Keep only last 100 alerts
    while (alertTableBody.children.length > 100) {
        alertTableBody.removeChild(alertTableBody.lastChild);
    }
}

function updateAlertSummary() {
    const count = alertTableBody.children.length;
    alertSummary.textContent = `Total: ${count} alerts`;
}

function updateCharts() {
    const alerts = getAlertsFromTable();
    
    // Update activity chart
    const activityCount = {};
    const severityCount = {};
    
    alerts.forEach(alert => {
        const activity = alert.activity || 'Unknown';
        activityCount[activity] = (activityCount[activity] || 0) + 1;
        
        const severity = alert.severity || 'low';
        severityCount[severity] = (severityCount[severity] || 0) + 1;
    });
    
    // Update stats chart
    const activities = ['walking', 'running', 'sitting', 'falling', 'climbing'];
    const data = activities.map(act => activityCount[act] || 0);
    statsChart.data.datasets[0].data = data;
    statsChart.update();
    
    // Update severity chart
    const severityData = [
        severityCount.high || 0,
        severityCount.medium || 0,
        severityCount.low || 0
    ];
    severityChart.data.datasets[0].data = severityData;
    severityChart.update();
}

function getAlertsFromTable() {
    const alerts = [];
    const rows = alertTableBody.querySelectorAll('tr');
    
    rows.forEach(row => {
        if (row.querySelector('td')) {
            const cells = row.querySelectorAll('td');
            alerts.push({
                timestamp: cells[1]?.textContent || '',
                severity: cells[2]?.textContent?.toLowerCase() || 'low',
                activity: cells[3]?.textContent || 'Unknown',
                message: cells[4]?.textContent || ''
            });
        }
    });
    
    return alerts;
}

// Event handlers
refreshBtn.addEventListener('click', fetchAlerts);

clearBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear all alerts?')) {
        alertTableBody.innerHTML = '';
        updateAlertSummary();
        updateCharts();
        showNotification('Alerts cleared', 'info');
    }
});

exportBtn.addEventListener('click', exportAlerts);

// Fetch alerts
function fetchAlerts() {
    fetch('/api/alerts')
        .then(response => response.json())
        .then(data => {
            alertTableBody.innerHTML = '';
            if (data.alerts && data.alerts.length > 0) {
                data.alerts.reverse().forEach(alert => addAlertToTable(alert));
                updateAlertSummary();
                updateCharts();
            } else {
                alertTableBody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center text-muted">No alerts found</td>
                    </tr>
                `;
            }
        })
        .catch(error => console.error('Error fetching alerts:', error));
}

function exportAlerts() {
    const alerts = getAlertsFromTable();
    
    if (alerts.length === 0) {
        showNotification('No alerts to export', 'warning');
        return;
    }
    
    // Create CSV
    const headers = ['Timestamp', 'Severity', 'Activity', 'Message'];
    const csvRows = [headers.join(',')];
    
    alerts.forEach(alert => {
        const row = [
            `"${alert.timestamp}"`,
            `"${alert.severity}"`,
            `"${alert.activity}"`,
            `"${alert.message}"`
        ];
        csvRows.push(row.join(','));
    });
    
    const csv = csvRows.join('\n');
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alerts_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
}

function showNotification(message, type = 'info') {
    // Simple notification
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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchAlerts();
    
    // Update every 30 seconds
    setInterval(fetchAlerts, 30000);
});