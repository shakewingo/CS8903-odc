'use client';

import { ChevronDown } from 'lucide-react';

export default function HeroSection() {
  return (
    <section
      id="hero"
      className="section-dark grain-overlay relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Topographic decorative lines */}
      <div className="absolute inset-0 opacity-[0.04] pointer-events-none">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          {[...Array(12)].map((_, i) => (
            <ellipse
              key={i}
              cx="55%"
              cy="50%"
              rx={180 + i * 60}
              ry={120 + i * 40}
              fill="none"
              stroke="currentColor"
              strokeWidth="1"
            />
          ))}
        </svg>
      </div>

      <div className="section-inner relative z-10 text-center max-w-4xl mx-auto">
        <h1 className="heading-display text-5xl md:text-7xl text-text-on-dark mb-6">
          RL-Driven Sustainable
          <br />
          <span className="text-accent-green-light">Land-Use Allocation</span>
        </h1>

        <p className="text-xl md:text-2xl text-text-on-dark-muted font-light leading-relaxed max-w-2xl mx-auto mb-4">
          Optimizing ecosystem service values in the Lake Malawi Basin
          through deep reinforcement learning
        </p>

        <div className="flex items-center justify-center gap-8 mt-12 text-sm text-text-on-dark-muted">
          <div className="flex flex-col items-center">
            <span className="text-2xl font-semibold text-text-on-dark">25 km</span>
            <span className="mt-1">Study radius</span>
          </div>
          <div className="w-px h-10 bg-border-dark" />
          <div className="flex flex-col items-center">
            <span className="text-2xl font-semibold text-text-on-dark">50×50</span>
            <span className="mt-1">Grid cells</span>
          </div>
          <div className="w-px h-10 bg-border-dark" />
          <div className="flex flex-col items-center">
            <span className="text-2xl font-semibold text-text-on-dark">9</span>
            <span className="mt-1">Land-use classes</span>
          </div>
        </div>
      </div>

      {/* Scroll cue */}
      <a
        href="#concepts"
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-text-on-dark-muted text-xs tracking-widest uppercase scroll-indicator"
      >
        <span>Explore</span>
        <ChevronDown size={20} />
      </a>
    </section>
  );
}
