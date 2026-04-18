'use client';

import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Rectangle, useMapEvents, useMap } from 'react-leaflet';
import type { LatLngBoundsExpression } from 'leaflet';
import { MapPin } from 'lucide-react';

const DEFAULT_CENTER: [number, number] = [-13.934564, 34.542859];
const DEFAULT_ZOOM = 9;
const HALF_SIDE_DEG = 25 / 2 / 111; // ~25km in degrees (rough)

interface MapDashboardProps {
  studyArea: { lat: number; lng: number } | null;
  onSelectArea: (coords: { lat: number; lng: number }) => void;
}

function ClickHandler({ onSelectArea }: { onSelectArea: (coords: { lat: number; lng: number }) => void }) {
  useMapEvents({
    click(e) {
      onSelectArea({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
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
  const bounds: LatLngBoundsExpression | null = studyArea
    ? [
        [studyArea.lat - HALF_SIDE_DEG, studyArea.lng - HALF_SIDE_DEG],
        [studyArea.lat + HALF_SIDE_DEG, studyArea.lng + HALF_SIDE_DEG],
      ]
    : null;

  return (
    <div className="relative w-full h-full">
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        className="w-full h-full z-0"
        scrollWheelZoom={true}
        style={{ background: '#0C1B2A' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <ClickHandler onSelectArea={onSelectArea} />
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
            <span className="text-text-on-dark-muted">Click map to select study area</span>
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
