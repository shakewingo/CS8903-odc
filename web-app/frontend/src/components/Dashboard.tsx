'use client';

import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  ReferenceLine
} from 'recharts';
import { Loader2 } from 'lucide-react';

interface LandCoverStats {
  [year: string]: {
    classes: {
      [id: string]: {
        label: string;
        percentage: number;
        color: string;
      }
    }
  }
}

interface ETStats {
  [year: string]: {
    mean: number;
    std: number;
    min: number;
    max: number;
    p25: number;
    p50: number;
    p75: number;
  }
}

interface ChartData {
  name: string; // Year
  [key: string]: string | number;
}

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [lcData, setLcData] = useState<ChartData[]>([]);
  const [combinedData, setCombinedData] = useState<ChartData[]>([]);
  const [colors, setColors] = useState<Record<string, string>>({});
  const [landCoverKeys, setLandCoverKeys] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [lcRes, etRes] = await Promise.all([
          fetch('/data/land_cover_stats.json'),
          fetch('/data/et_stats.json')
        ]);

        if (!lcRes.ok || !etRes.ok) throw new Error('Failed to fetch data');

        const lcStats: LandCoverStats = await lcRes.json();
        const etStats: ETStats = await etRes.json();

        // Process Land Cover Data
        const lcProcessed: ChartData[] = [];
        const colorMap: Record<string, string> = {};
        const allKeys = new Set<string>();

        Object.keys(lcStats).sort().forEach(year => {
          const row: ChartData = { name: year };
          const yearData = lcStats[year];
          
          Object.values(yearData.classes).forEach(cls => {
            row[cls.label] = cls.percentage;
            colorMap[cls.label] = cls.color;
            allKeys.add(cls.label);
          });
          lcProcessed.push(row);
        });

        setLcData(lcProcessed);
        setColors(colorMap);
        setLandCoverKeys(Array.from(allKeys));

        // Process Combined Data (ET + Vegetation)
        const combined: ChartData[] = [];
        const years = Object.keys(etStats).sort();

        years.forEach(year => {
          const et = etStats[year];
          // Find corresponding LC data
          const lcRow = lcProcessed.find(d => d.name === year) || { name: year };
          
          // Calculate Total Vegetated Land
          const crops = (lcRow['Crops'] as number) || 0;
          const trees = (lcRow['Trees'] as number) || 0;
          const rangeland = (lcRow['Rangeland'] as number) || 0;
          const vegetated = crops + trees + rangeland;

          combined.push({
            Mean_ET: et.mean,
            P25_ET: et.p25,
            P75_ET: et.p75,
            Crops: crops,
            Trees: trees,
            Rangeland: rangeland,
            Vegetated_Land: vegetated,
            ...lcRow
          });
        });

        setCombinedData(combined);
        setLoading(false);

      } catch (err) {
        console.error(err);
        setError('Failed to load analysis data.');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div className="flex h-96 items-center justify-center"><Loader2 className="animate-spin h-8 w-8 text-blue-500" /></div>;
  if (error) return <div className="text-red-500 text-center p-8">{error}</div>;

  return (
    <div className="space-y-12 p-4 md:p-8 max-w-7xl mx-auto">
      <div className="space-y-4">
        <h1 className="text-3xl font-bold text-gray-900">RL-Driven Sustainable Land-Use Allocation</h1>
        <p className="text-gray-600 max-w-3xl">
          This dashboard displays the exploratory data analysis for the research on sustainable land-use allocation 
          in the Lake Malawi Basin. It visualizes the relationship between land cover changes 
          and evapotranspiration (ET) levels over the years 2017-2024.
        </p>
      </div>

      {/* Plot 1: Land Cover Composition */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h2 className="text-xl font-semibold mb-6">1. Land Cover Composition (2017-2024)</h2>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={lcData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis label={{ value: 'Percentage Coverage (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value: number | undefined) => [`${Number(value || 0).toFixed(1)}%`, '']}
                contentStyle={{ borderRadius: '8px' }}
              />
              <Legend />
              {landCoverKeys.map(key => (
                <Bar key={key} dataKey={key} stackId="a" fill={colors[key] || '#8884d8'} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Plot 2: ET vs Vegetation */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h2 className="text-xl font-semibold mb-6">2. Evapotranspiration vs. Vegetated Land Cover</h2>
        <div className="h-[500px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={combinedData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              
              {/* Left Axis: ET */}
              <YAxis 
                yAxisId="left" 
                label={{ value: 'Mean ET (kg/m²/year)', angle: -90, position: 'insideLeft', offset: 0, fill: '#3b82f6' }} 
                tick={{ fill: '#3b82f6' }}
                domain={['auto', 'auto']}
              />
              
              {/* Right Axis: Land Cover % */}
              <YAxis 
                yAxisId="right" 
                orientation="right" 
                label={{ value: 'Land Cover Percentage (%)', angle: 90, position: 'insideRight', offset: 0 }} 
                domain={['auto', 'auto']}
              />

              <Tooltip 
                contentStyle={{ borderRadius: '8px' }}
                formatter={(value: number | undefined, name: string | undefined) => {
                  const val = Number(value || 0);
                  const n = String(name || '');
                  if (n.includes('ET')) return [`${val.toFixed(1)} kg/m²`, n];
                  return [`${val.toFixed(1)}%`, n];
                }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />

              {/* ET Area (Range) */}
              <Area 
                yAxisId="left"
                type="monotone" 
                dataKey="P75_ET" 
                stroke="none" 
                fill="#3b82f6" 
                fillOpacity={0.1} 
                className='hidden-legend' /* Hack to hide from legend if needed, but composed chart handles it okay usually */
              />
              {/* We want to show a band, Recharts area is usually 0 to value. 
                  To do a true band (P25 to P75), we need a custom shape or two areas overlaid. 
                  For simplicity, let's just show Mean and Lines for now as Area range is complex in Recharts without processed data tuples. 
              */}
              
              <Line yAxisId="left" type="monotone" dataKey="Mean_ET" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} name="Mean ET" />
              
              <Line yAxisId="right" type="monotone" dataKey="Crops" stroke={colors['Crops'] || "orange"} strokeDasharray="5 5" name="Crops %" />
              <Line yAxisId="right" type="monotone" dataKey="Trees" stroke={colors['Trees'] || "green"} strokeDasharray="5 5" name="Trees %" />
              <Line yAxisId="right" type="monotone" dataKey="Rangeland" stroke={colors['Rangeland'] || "#e3e2c3"} strokeDasharray="5 5" name="Rangeland %" />
              <Line yAxisId="right" type="monotone" dataKey="Vegetated_Land" stroke="darkgreen" strokeWidth={2} name="Total Vegetated Land %" />

            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        <div className="mt-8 bg-gray-50 p-6 rounded-lg text-sm text-gray-700 leading-relaxed">
          <h3 className="font-semibold text-gray-900 mb-2">Conclusion</h3>
          <p className="mb-2">
            Basically, the mean <strong>ET</strong> has gradually decreased from 2017 to 2024, with some fluctuations during 2019 to 2022. 
            This trend shows a <strong>negative correlation</strong> with the gradual increase of crop land and a <strong>positive correlation</strong> with the gradual decrease of total vegetated land.
          </p>
          <p className="mb-2">
            There is also an obvious opposite change between trees and rangeland, which indicates a <strong>potential substitution</strong> for each other during the development period.
          </p>
          <p className="text-gray-500 italic">
            *Note: We did not calculate exact statistics like Pearson correlation as there are only annual data points available.*
          </p>
        </div>
      </div>
    </div>
  );
}
