'use client';

import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Rectangle, GeoJSON, useMapEvents, useMap } from 'react-leaflet';
import type { LatLngBoundsExpression } from 'leaflet';
import L from 'leaflet';
import { MapPin } from 'lucide-react';

const LAKE_CENTER: [number, number] = [-13.934564, 34.542859];
const DEFAULT_ZOOM = 9;
const HALF_SIDE_DEG = 25 / 2 / 111; // ~25km box in degrees (rough)

interface MapDashboardProps {
  studyArea: { lat: number; lng: number } | null;
  onSelectArea: (coords: { lat: number; lng: number }) => void;
}

function ClickHandler({ onSelectArea, bufferLayer }: {
  onSelectArea: (coords: { lat: number; lng: number }) => void;
  bufferLayer: L.GeoJSON | null;
}) {
  useMapEvents({
    click(e) {
      // Only allow clicks inside the buffer polygon
      if (bufferLayer) {
        const point = e.latlng;
        let inside = false;
        bufferLayer.eachLayer((layer) => {
          if ((layer as L.Polygon).getBounds && (layer as L.Polygon).getBounds().contains(point)) {
            // More precise check using leaflet-pip-like logic
            const poly = layer as L.Polygon;
            const latlngs = poly.getLatLngs()[0] as L.LatLng[];
            if (isPointInPolygon(point, latlngs)) {
              inside = true;
            }
          }
        });
        if (inside) {
          onSelectArea({ lat: e.latlng.lat, lng: e.latlng.lng });
        }
      }
    },
  });
  return null;
}

// Ray-casting point-in-polygon check
function isPointInPolygon(point: L.LatLng, polygon: L.LatLng[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].lat, yi = polygon[i].lng;
    const xj = polygon[j].lat, yj = polygon[j].lng;
    const intersect = ((yi > point.lng) !== (yj > point.lng)) &&
      (point.lat < (xj - xi) * (point.lng - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function FlyToArea({ lat, lng }: { lat: number; lng: number }) {
  const map = useMap();
  const prevRef = useRef<string>('');
  useEffect(() => {
    const key = `${lat},${lng}`;
    if (key !== prevRef.current) {
      prevRef.current = key;
      map.flyTo([lat, lng], 10, { duration: 0.8 });
    }
  }, [lat, lng, map]);
  return null;
}

export default function MapDashboard({ studyArea, onSelectArea }: MapDashboardProps) {
  const [bufferGeoJson, setBufferGeoJson] = useState<GeoJSON.Feature | null>(null);
  const [bufferLayerRef, setBufferLayerRef] = useState<L.GeoJSON | null>(null);

  useEffect(() => {
    fetch('/data/lake_malawi_expanded_25km.json')
      .then((r) => r.json())
      .then((data) => setBufferGeoJson(data))
      .catch(() => console.warn('Failed to load lake buffer GeoJSON'));
  }, []);

  const bounds: LatLngBoundsExpression | null = studyArea
    ? [
        [studyArea.lat - HALF_SIDE_DEG, studyArea.lng - HALF_SIDE_DEG],
        [studyArea.lat + HALF_SIDE_DEG, studyArea.lng + HALF_SIDE_DEG],
      ]
    : null;

  return (
    <div className="relative w-full h-full">
      <MapContainer
        center={LAKE_CENTER}
        zoom={DEFAULT_ZOOM}
        className="w-full h-full z-0"
        scrollWheelZoom={true}
        style={{ background: '#0C1B2A' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Lake Malawi 25km buffer zone boundary */}
        {bufferGeoJson && (
          <GeoJSON
            data={bufferGeoJson}
            ref={(ref) => { if (ref) setBufferLayerRef(ref); }}
            style={{
              color: '#1B6B4A',
              weight: 2,
              dashArray: '6 4',
              fillColor: '#1B6B4A',
              fillOpacity: 0.05,
            }}
          />
        )}

        <ClickHandler onSelectArea={onSelectArea} bufferLayer={bufferLayerRef} />
        {studyArea && <FlyToArea lat={studyArea.lat} lng={studyArea.lng} />}
        {bounds && (
          <Rectangle
            bounds={bounds}
            pathOptions={{
              color: '#1B6B4A',
              weight: 2,
              fillColor: '#1B6B4A',
              fillOpacity: 0.15,
            }}
          />
        )}
      </MapContainer>

      {/* Coordinate overlay */}
      <div className="absolute top-3 left-3 z-[1000] bg-bg-dark/85 backdrop-blur-sm rounded-lg px-3 py-2 border border-border-dark">
        <div className="flex items-center gap-2 text-xs text-text-on-dark">
          <MapPin size={12} className="text-accent-green" />
          {studyArea ? (
            <span className="font-mono">
              {studyArea.lat.toFixed(4)}, {studyArea.lng.toFixed(4)}
            </span>
          ) : (
            <span className="text-text-on-dark-muted">Click within the boundary to select study area</span>
          )}
        </div>
      </div>

      {/* 25km badge */}
      {studyArea && (
        <div className="absolute top-3 right-3 z-[1000] bg-accent-green/90 text-white text-[10px] font-semibold px-2 py-1 rounded">
          25km x 25km
        </div>
      )}
    </div>
  );
}
