'use client';

import { Globe, Satellite, BarChart3, ExternalLink } from 'lucide-react';

const DATA_SOURCES = [
  {
    icon: Satellite,
    title: 'Sentinel-2 Land Cover',
    desc: 'ESA 10m resolution satellite imagery (2024), downsampled to 10% for land cover classification. Provides 9-class land cover maps derived from the ESA WorldCover product.',
    tag: 'Land Cover',
    year: '2024',
    link: 'https://climat.esri.ca/datasets/esri::sentinel-2-land-cover-explorer',
  },
  {
    icon: Globe,
    title: 'MODIS Evapotranspiration',
    desc: 'NASA Moderate Resolution Imaging Spectroradiometer (MODIS) providing evapotranspiration (ET) data at regional scale via Microsoft Planetary Computer.',
    tag: 'ET Data',
    year: '2024',
    link: 'https://planetarycomputer.microsoft.com/explore?c=25.2044%2C-5.0249&z=5.62&v=2',
  },
  {
    icon: BarChart3,
    title: 'ESVD / Costanza et al.',
    desc: 'Ecosystem Services Value Database from Costanza et al. (2014), "Changes in the global value of ecosystem services." Calibrated to Lake Malawi using Zuze 2013 wetland anchor ($554/ha/yr).',
    tag: 'ESV Reference',
    year: '2014',
    link: 'https://www.sciencedirect.com/science/article/abs/pii/S0959378014000685',
  },
];

export default function ConceptsSection() {
  return (
    <section id="concepts" className="section-cream-compact">
      <div className="section-inner">
        <p className="heading-sub mb-2">Background</p>
        <h2 className="heading-section text-text-primary">
          Research Concepts
        </h2>
        <p className="text-text-secondary max-w-2xl mb-8 leading-relaxed">
          This study applies deep reinforcement learning to optimize land-use
          allocation around Lake Malawi, maximizing ecosystem service values
          while respecting ecological constraints such as water zone buffering
          and habitat contiguity.
        </p>

        {/* Data Sources */}
        <div className="grid md:grid-cols-3 gap-6">
          {DATA_SOURCES.map((src) => (
            <div
              key={src.title}
              className="bg-bg-card rounded-xl p-6 border border-border-warm hover:shadow-lg transition-shadow"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-accent-green/10 flex items-center justify-center">
                  <src.icon size={20} className="text-accent-green" />
                </div>
                <div>
                  <h3 className="font-semibold text-text-primary">{src.title}</h3>
                  <span className="text-xs text-accent-green font-medium">{src.tag} · {src.year}</span>
                </div>
              </div>
              <p className="text-sm text-text-secondary leading-relaxed mb-3">
                {src.desc}
              </p>
              <a
                href={src.link}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-xs font-medium text-accent-green hover:text-accent-green-light transition-colors"
              >
                View data source
                <ExternalLink size={12} />
              </a>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
