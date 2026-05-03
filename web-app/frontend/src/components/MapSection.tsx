'use client';

import { useState } from 'react';
import { MapPin, Search, ZoomIn, ZoomOut, Crosshair } from 'lucide-react';

const DEFAULT_LAT = '-13.934564';
const DEFAULT_LNG = '34.542859';

export default function MapSection() {
  const [lat, setLat] = useState(DEFAULT_LAT);
  const [lng, setLng] = useState(DEFAULT_LNG);
  const [locked, setLocked] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLocked(true);
  };

  const handleReset = () => {
    setLocked(false);
    setLat(DEFAULT_LAT);
    setLng(DEFAULT_LNG);
  };

  return (
    <section id="map" className="section-light">
      <div className="section-inner">
        <p className="heading-sub mb-2">Study Area</p>
        <h2 className="heading-section text-text-primary">
          Interactive Map
        </h2>
        <p className="text-text-secondary max-w-2xl mb-10 leading-relaxed">
          Explore Lake Malawi and the surrounding 25km area. Enter coordinates
          to define a study region, which represent a 25km×25km area and will be
          divided into a 50×50 grid for analysis.
        </p>

        <div className="grid lg:grid-cols-[1fr_320px] gap-8">
          {/* Map Area */}
          <div className="map-placeholder aspect-[4/3] lg:aspect-auto lg:min-h-[520px] relative">
            {/* Simulated lake shape */}
            <div className="absolute inset-0 flex items-center justify-center">
              <svg
                viewBox="0 0 400 500"
                className="w-3/4 h-3/4 opacity-30"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M200,30 C230,30 270,60 280,100 C290,140 285,180 275,220 C265,260 260,300 265,340 C270,380 260,420 240,450 C220,470 190,480 170,460 C150,440 145,400 150,360 C155,320 150,280 140,240 C130,200 125,160 135,120 C145,80 170,30 200,30Z"
                  fill="#419bdf"
                  stroke="#2E86AB"
                  strokeWidth="2"
                />
              </svg>
            </div>

            {/* Grid overlay when area is selected */}
            {locked && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-40 h-40 border-2 border-accent-green rounded-sm opacity-80">
                <div
                  className="w-full h-full"
                  style={{
                    backgroundImage:
                      'linear-gradient(rgba(27,107,74,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(27,107,74,0.3) 1px, transparent 1px)',
                    backgroundSize: '10% 10%',
                  }}
                />
                <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-accent-green text-white text-xs px-2 py-0.5 rounded whitespace-nowrap">
                  25km × 25km
                </div>
              </div>
            )}

            {/* Zoom controls */}
            <div className="absolute top-4 right-4 flex flex-col gap-1">
              <button className="w-9 h-9 bg-white/90 backdrop-blur rounded-lg flex items-center justify-center shadow-md hover:bg-white">
                <ZoomIn size={16} className="text-text-primary" />
              </button>
              <button className="w-9 h-9 bg-white/90 backdrop-blur rounded-lg flex items-center justify-center shadow-md hover:bg-white">
                <ZoomOut size={16} className="text-text-primary" />
              </button>
            </div>

            {/* Attribution placeholder */}
            <div className="absolute bottom-3 left-3 text-[10px] text-white/60 bg-black/20 px-2 py-1 rounded">
              Sentinel-2 · 2024 · 10% sample
            </div>
          </div>

          {/* Coordinate Input Sidebar */}
          <div className="space-y-6">
            <div className="bg-bg-card rounded-xl border border-border-warm p-6">
              <div className="flex items-center gap-2 mb-5">
                <Crosshair size={18} className="text-accent-green" />
                <h3 className="font-semibold text-text-primary">
                  Coordinates
                </h3>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-text-muted mb-1.5 uppercase tracking-wider">
                    Latitude
                  </label>
                  <input
                    type="text"
                    value={lat}
                    onChange={(e) => setLat(e.target.value)}
                    disabled={locked}
                    className="w-full px-3 py-2.5 rounded-lg border border-border-warm bg-bg-warm text-sm font-mono text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-green/30 focus:border-accent-green disabled:opacity-60"
                    placeholder="-13.934564"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-text-muted mb-1.5 uppercase tracking-wider">
                    Longitude
                  </label>
                  <input
                    type="text"
                    value={lng}
                    onChange={(e) => setLng(e.target.value)}
                    disabled={locked}
                    className="w-full px-3 py-2.5 rounded-lg border border-border-warm bg-bg-warm text-sm font-mono text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-green/30 focus:border-accent-green disabled:opacity-60"
                    placeholder="34.542859"
                  />
                </div>

                {!locked ? (
                  <button
                    type="submit"
                    className="w-full py-2.5 rounded-lg bg-accent-green text-white font-medium text-sm hover:bg-accent-green/90 transition-colors flex items-center justify-center gap-2"
                  >
                    <Search size={14} />
                    Set Study Area
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleReset}
                    className="w-full py-2.5 rounded-lg border-2 border-accent-green text-accent-green font-medium text-sm hover:bg-accent-green/5 transition-colors"
                  >
                    Reset Area
                  </button>
                )}
              </form>
            </div>

            {/* Area info card */}
            {locked && (
              <div className="bg-accent-green/5 border border-accent-green/20 rounded-xl p-5">
                <div className="flex items-start gap-3">
                  <MapPin size={18} className="text-accent-green mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-semibold text-text-primary mb-1">
                      Study Area Selected
                    </p>
                    <p className="text-xs text-text-secondary leading-relaxed">
                      Center: {lat}, {lng}
                      <br />
                      Coverage: 25km × 25km
                      <br />
                      Grid: 50 × 50 cells (2,500 total)
                      <br />
                      Cell size: 5 × 5 pixels
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
