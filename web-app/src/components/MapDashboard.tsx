'use client';

import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Rectangle, Circle, useMapEvents, useMap } from 'react-leaflet';
import type { LatLngBoundsExpression } from 'leaflet';
import { MapPin } from 'lucide-react';

// Lake Malawi approximate center and the 25km buffer radius for the study zone
const LAKE_CENTER: [number, number] = [-13.934564, 34.542859];
const BUFFER_RADIUS_KM = 25;
const BUFFER_RADIUS_M = BUFFER_RADIUS_KM * 1000;

const DEFAULT_ZOOM = 9;
const HALF_SIDE_DEG = 25 / 2 / 111; // ~25km box in degrees (rough)

// Check if a point is within the allowed buffer zone
function isWithinBuffer(lat: number, lng: number): boolean {
  const R = 6371; // Earth radius in km
  const dLat = (lat - LAKE_CENTER[0]) * Math.PI / 180;
  const dLng = (lng - LAKE_CENTER[1]) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(LAKE_CENTER[0] * Math.PI / 180) * Math.cos(lat * Math.PI / 180) *
    Math.sin(dLng / 2) ** 2;
  const dist = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return dist <= BUFFER_RADIUS_KM;
}

interface MapDashboardProps {
  studyArea: { lat: number; lng: number } | null;
  onSelectArea: (coords: { lat: number; lng: number }) => void;
}

function ClickHandler({ onSelectArea }: { onSelectArea: (coords: { lat: number; lng: number }) => void }) {
  useMapEvents({
    click(e) {
      if (isWithinBuffer(e.latlng.lat, e.latlng.lng)) {
        onSelectArea({ lat: e.latlng.lat, lng: e.latlng.lng });
      }
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

        {/* 25km buffer zone boundary — selectable area */}
        <Circle
          center={LAKE_CENTER}
          radius={BUFFER_RADIUS_M}
          pathOptions={{
            color: '#1B6B4A',
            weight: 1.5,
            dashArray: '6 4',
            fillColor: 'transparent',
            fillOpacity: 0,
          }}
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

      {/* Grey overlay mask — covers area outside the buffer zone via CSS */}
      <div className="absolute inset-0 z-[400] pointer-events-none overflow-hidden">
        <div
          className="absolute rounded-full border-[3000px] border-black/40"
          style={{
            width: '0px',
            height: '0px',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            /* The circle mask is approximate — Leaflet handles the real boundary via Circle */
            display: 'none',
          }}
        />
      </div>

      {/* Coordinate overlay */}
      <div className="absolute top-3 left-3 z-[1000] bg-bg-dark/85 backdrop-blur-sm rounded-lg px-3 py-2 border border-border-dark">
        <div className="flex items-center gap-2 text-xs text-text-on-dark">
          <MapPin size={12} className="text-accent-green" />
          {studyArea ? (
            <span className="font-mono">
              {studyArea.lat.toFixed(4)}, {studyArea.lng.toFixed(4)}
            </span>
          ) : (
            <span className="text-text-on-dark-muted">Click within the green boundary to select study area</span>
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
