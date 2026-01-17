// GazaMapVisualizer.js
// Interactive map for Gaza humanitarian sites

class GazaMapVisualizer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.options = {
            center: [31.5, 34.466],
            zoom: 11,
            tileLayer: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attribution: '© OpenStreetMap',
            ...options
        };
        this.map = null;
        this.sites = [];
        this.selectedSites = [];
        this.markers = new Map();
    }

    init() {
        this.map = L.map(this.containerId).setView(this.options.center, this.options.zoom);
        L.tileLayer(this.options.tileLayer, { attribution: this.options.attribution, maxZoom: 18 }).addTo(this.map);
        this.addControls();
        return this;
    }

    addControls() {
        L.control.scale({ imperial: false }).addTo(this.map);
        L.control.fullscreen({ position: 'topright' }).addTo(this.map);
        const layers = {
            "Base": L.tileLayer(this.options.tileLayer),
            "Satellite": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')
        };
        L.control.layers(layers).addTo(this.map);
        return this;
    }

    async loadData(url) {
        try {
            const res = await fetch(url);
            this.sites = await res.json();
        } catch {
            this.loadSampleData();
        }
        this.plotSites();
        return this.sites;
    }

    loadSampleData() {
        this.sites = [
            { id: 1, name: "Al-Shifa Hospital", lat: 31.5247, lng: 34.4563, type: "hospital", risk: 0.7, access: 0.6, priority: 0.9, score: 0.85 },
            { id: 2, name: "Jabalia Camp", lat: 31.5385, lng: 34.4909, type: "camp", risk: 0.8, access: 0.4, priority: 0.9, score: 0.72 },
            { id: 3, name: "Al-Azhar Univ", lat: 31.5144, lng: 34.4583, type: "school", risk: 0.4, access: 0.7, priority: 0.7, score: 0.68 }
        ];
        return this.sites;
    }

    plotSites() {
        this.clearMarkers();
        this.sites.forEach(s => this.addMarker(s));
        if (this.sites.length) {
            const bounds = this.sites.map(s => [s.lat, s.lng]);
            this.map.fitBounds(bounds, { padding: [50, 50] });
        }
    }

    addMarker(site) {
        const color = site.risk > 0.7 ? '#e74c3c' : site.risk > 0.4 ? '#f39c12' : '#2ecc71';
        const size = site.priority > 0.8 ? 12 : site.priority > 0.6 ? 10 : 8;
        const marker = L.circleMarker([site.lat, site.lng], { radius: size, fillColor: color, color: '#fff', weight: 2, fillOpacity: 0.8 })
            .bindPopup(this.createPopup(site), { maxWidth: 300 });
        marker.on('click', () => this.onClick(site));
        marker.addTo(this.map);
        this.markers.set(site.id, marker);
    }

    createPopup(site) {
        return `
            <div style="font-family:sans-serif">
                <h3>${site.name}</h3>
                <p>Type: ${site.type} | Score: ${site.score.toFixed(2)}</p>
                <p>Risk: ${Math.round(site.risk*100)}% | Access: ${Math.round(site.access*100)}%</p>
                <button onclick="window.selectSite(${site.id})">Select</button>
            </div>
        `;
    }

    onClick(site) {
        this.displayInfo(site);
        if (typeof window.onSiteSelected === 'function') window.onSiteSelected(site);
    }

    displayInfo(site) {
        const div = document.getElementById('site-info');
        if (div) div.innerHTML = `<h3>${site.name}</h3><p>Score: ${site.score.toFixed(2)}</p>`;
    }

    selectSite(id) {
        const site = this.sites.find(s => s.id === id);
        if (!site || this.selectedSites.includes(site)) return;
        this.selectedSites.push(site);
        const marker = this.markers.get(id);
        if (marker) marker.setStyle({ fillColor: '#3498db', weight: 3 });
        this.updateSelectedDisplay();
    }

    updateSelectedDisplay() {
        const container = document.getElementById('selected-sites');
        if (!container) return;
        container.innerHTML = `<h3>Selected (${this.selectedSites.length})</h3>` +
            this.selectedSites.map(s => `<p>${s.name} | Score: ${s.score.toFixed(2)}</p>`).join('') +
            (this.selectedSites.length ? `<button onclick="window.exportSites()">Save</button>` : '');
    }

    clearMarkers() { this.markers.forEach(m => this.map.removeLayer(m)); this.markers.clear(); }

    filter(criteria) {
        const filtered = this.sites.filter(s =>
            (!criteria.region || s.region === criteria.region) &&
            (!criteria.minScore || s.score >= criteria.minScore) &&
            (!criteria.maxRisk || s.risk <= criteria.maxRisk)
        );
        this.clearMarkers();
        filtered.forEach(s => this.addMarker(s));
        return filtered;
    }

    exportSites() {
        const blob = new Blob([JSON.stringify(this.selectedSites, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'selected_sites.json';
        link.click();
    }
}

// Global functions
window.GazaMapVisualizer = GazaMapVisualizer;
window.selectSite = id => window.mapInstance?.selectSite(id);
window.exportSites = () => window.mapInstance?.exportSites();
window.onSiteSelected = site => console.log('Site selected:', site.name);

window.initGazaMap = (container = 'map', url = 'data/sites.json') => {
    const viz = new GazaMapVisualizer(container).init();
    viz.loadData(url);
    window.mapInstance = viz;
    return viz;
};

document.addEventListener('DOMContentLoaded', () => { if (document.getElementById('map')) window.initGazaMap(); });
