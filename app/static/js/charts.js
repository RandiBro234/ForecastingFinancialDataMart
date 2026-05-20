// Common Chart Options
const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'bottom',
            labels: {
                usePointStyle: true,
                padding: 20,
                font: {
                    family: "'Inter', sans-serif"
                }
            }
        },
        tooltip: {
            backgroundColor: '#111827',
            titleFont: { family: "'Inter', sans-serif", size: 13 },
            bodyFont: { family: "'Inter', sans-serif", size: 13 },
            padding: 10,
            cornerRadius: 8,
            callbacks: {
                label: function(context) {
                    let label = context.dataset.label || '';
                    if (label) {
                        label += ': ';
                    }
                    if (context.parsed.y !== null) {
                        label += context.parsed.y + ' B';
                    }
                    return label;
                }
            }
        }
    },
    scales: {
        x: {
            grid: { display: false },
            ticks: { font: { family: "'Inter', sans-serif" } }
        },
        y: {
            grid: { borderDash: [5, 5], color: '#e5e7eb' },
            ticks: {
                font: { family: "'Inter', sans-serif" },
                callback: function(value) {
                    return value + ' B';
                }
            }
        }
    }
};

document.addEventListener('DOMContentLoaded', function() {
    // 1. Dashboard Chart
    const trendCtx = document.getElementById('revenueTrendChart');
    if (trendCtx) {
        fetch('/api/revenue-trend')
            .then(res => res.json())
            .then(data => {
                new Chart(trendCtx, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            label: 'Actual Revenue',
                            data: data.revenue,
                            borderColor: '#4F46E5',
                            backgroundColor: 'rgba(79, 70, 229, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0
                        }]
                    },
                    options: commonOptions
                });
            });
    }

    // 2. Forecast Result Chart
    const forecastCtx = document.getElementById('forecastChart');
    if (forecastCtx) {
        fetch('/api/forecast')
            .then(res => res.json())
            .then(data => {
                new Chart(forecastCtx, {
                    type: 'line',
                    data: {
                        // combine labels (actuals and forecasts)
                        labels: [...data.actual_labels, ...data.forecast_labels],
                        datasets: [
                            {
                                label: 'Actual Revenue',
                                // pad the end with nulls
                                data: [...data.actual, ...Array(data.forecast_labels.length).fill(null)],
                                borderColor: '#111827',
                                borderWidth: 2,
                                tension: 0.1,
                                pointRadius: 0
                            },
                            {
                                label: 'SARIMA Forecast',
                                // pad the start with nulls
                                data: [...Array(data.actual_labels.length).fill(null), ...data.sarima],
                                borderColor: '#22C55E',
                                borderWidth: 3,
                                borderDash: [5, 5],
                                tension: 0.1,
                                pointRadius: 2
                            },
                            {
                                label: 'GPR Forecast',
                                data: [...Array(data.actual_labels.length).fill(null), ...data.gpr],
                                borderColor: '#4F46E5',
                                borderWidth: 2,
                                borderDash: [3, 3],
                                tension: 0.1,
                                pointRadius: 0
                            },
                            {
                                label: 'GPR Upper Bound',
                                data: [...Array(data.actual_labels.length).fill(null), ...data.gpr_upper],
                                borderColor: 'rgba(99, 102, 241, 0.2)',
                                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                                borderWidth: 1,
                                fill: '+1',
                                pointRadius: 0,
                                tension: 0.1
                            },
                            {
                                label: 'GPR Lower Bound',
                                data: [...Array(data.actual_labels.length).fill(null), ...data.gpr_lower],
                                borderColor: 'rgba(99, 102, 241, 0.2)',
                                backgroundColor: 'transparent',
                                borderWidth: 1,
                                fill: false,
                                pointRadius: 0,
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        ...commonOptions,
                        interaction: { mode: 'index', intersect: false }
                    }
                });
            });
    }

    // 3. Evaluation Chart
    const evalCtx = document.getElementById('evaluationChart');
    if (evalCtx) {
        fetch('/api/evaluation')
            .then(res => res.json())
            .then(data => {
                new Chart(evalCtx, {
                    type: 'bar',
                    data: {
                        labels: ['MAE', 'RMSE'],
                        datasets: [
                            {
                                label: 'SARIMA',
                                data: [data.sarima.MAE, data.sarima.RMSE],
                                backgroundColor: '#22C55E',
                                borderRadius: 4
                            },
                            {
                                label: 'GPR',
                                data: [data.gpr.MAE, data.gpr.RMSE],
                                backgroundColor: '#4F46E5',
                                borderRadius: 4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return context.dataset.label + ': ' + context.parsed.y.toLocaleString();
                                    }
                                }
                            }
                        }
                    }
                });
            });
    }
});