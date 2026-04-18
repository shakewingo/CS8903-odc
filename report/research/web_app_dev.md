The PWA web-app is for the purpose of displaying the main findings in my research

## Research Overview
./report/research/final_paper_writing.md

## Requirements
1. Continue the development of the web-app: https://cs-8903-odc.vercel.app/. The current content should be mostly removed as they are out of date.
2. Pick the tech stack that is light and easy for me to read and maintain later. Try to keep the layout as concise as possible, avoid heavy layout changes in the future. It needs to be a PWA web-app.
3. My rough idea of the web-app is to have main sections below:
- Concept introduction
    - Some concepts in this research may be introduced at beginning of the web-app, e.g. research purpose, study case, data source (e.g. sentinel-2, MODIS, ESVD etc.), land-use type
- An interactive map of Lake Malawi with 25km area surrounding
    - The map should load 2024 Sentinel-2 raw data which is downsampled to 10% for its land cover type
    - It should allow zoom-in and zoom-out with low latency. Also aside of th   e map, there's a coordinates box for user to input, e.g. -13.934564, 34.542859, when user hits enter, the map will zoom in to a square area that is a 25km * 25km area centered at the given coordinates. This is jus t the area that I used for my research. 
- The study area ecological value heatmap
    - Once user chosen the study area, this section can display a heatmap of the ecological value, just like what is shown in the paper. "Each cell displays horizontal bars representing land-use type fractions, and the background color encodes the cell’s total ecosystem value (red = high,white = low)". It is understandable that creating the heatmap may take some time, so the loading hints may be needed.
    - Additionally, the ESV value used to generate the heatmap is also interactive. By default, the value is used as the one in ./src/config.py but allowing user to put their own values. Note that the reference and calibration source of the default ESV value, for example from what paper/articles, should be mentioned in a small question mark icon to better guide users. It should also mention that the default value of crop is non-regenerative and 35% increasing is suggested for regenerative scenario.
- Load land-allocation model engine 
    - We need to split this part to 2 phase: 1) Load static training model that has fixed config values for inference area, i.e. load the model used for exp I, II and III in the research and block user to change the config values on their own, maybe have a button to switch between different experiments; 2) Retrain the model and load for inference with user-defined config values. The training process usually takes ~15mins, so should have hints of waiting, but more than that, it's about model saving and caching system to design if choosing to develop this feature.
    - In current phase, we can only consider 1)

- Visualize the optimization result
    - Once the land-allocation engine is loaded, the optimization result will be visualized in a similar way as the heatmap, plus highlighted box to indicate which cell changesd. The frontend design needs to be friendly and straightforward for users to compare the difference.
    - A table that ranked top 50 cells that have ESV change from high to low and corresponds with what land-use type change can be displayed in later section

4. Document web-app design status regularly

---

## Initial Design & Implementation (Demo Phase)

### Tech Stack
| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | Next.js 16 (existing) | Already set up, PWA-ready via `@ducanh2912/next-pwa` |
| Styling | Tailwind CSS v4 (existing) | Lightweight, maintainable utility classes |
| Icons | lucide-react (existing) | Consistent icon set, tree-shakeable |
| Map | Leaflet + react-leaflet (planned) | Free, no API key, lightweight. Placeholder in demo |
| Fonts | Crimson Pro (display) + Manrope (body) | Distinctive serif/sans pairing with scientific editorial tone |

No new dependencies were added for the demo phase.

### Design Direction: "Earth-Science Editorial"
- **Light sections**: warm cream `#FAF7F2` for text-heavy content (Concepts, Map, Model)
- **Dark sections**: deep navy `#0C1B2A` for data visualizations (Hero, Heatmap, Results)
- **Accent palette**: forest green `#1B6B4A` (primary), gold `#C49A2A` (secondary), lake blue `#2E86AB`
- **Land cover colors**: directly from `config.py` for consistency with paper figures
- **Side navigation**: fixed dot nav on the right edge, scroll-aware active state

### Page Structure (6 sections, top-to-bottom scroll)

#### 1. Hero (`HeroSection.tsx`) — dark bg
- Full viewport height, centered title (no "CS 8903" subtitle)
- Subtitle: "Optimizing ecosystem service values in the Lake Malawi Basin through deep reinforcement learning"
- Key stats row: 25km radius, 50×50 grid, 9 land-use classes
- Scroll cue at bottom

#### 2. Concepts (`ConceptsSection.tsx`) — cream bg
- 3 data-source cards with external links and year:
  - Sentinel-2 Land Cover (2024) → Esri Land Cover Explorer
  - MODIS Evapotranspiration (2024) → Microsoft Planetary Computer
  - ESVD / Costanza et al. (2014) → ScienceDirect paper
- Land cover classification section removed (per feedback: lacked information)

#### 3. Interactive Map (`MapSection.tsx`) — light bg
- Left: map area (placeholder in demo; Leaflet planned for real data)
- Right sidebar: coordinate input (lat/lng), "Set Study Area" button
- Updated description: "...which represent a 25km×25km area and will be divided into a 50×50 grid..."
- On submit: shows 25km×25km grid overlay on map + area info card
- Zoom controls (placeholder)

#### 4. ESV Heatmap (`HeatmapSection.tsx`) — dark bg
- Left: 10×10 grid of heatmap cells (50×50 in production)
  - Each cell: background color encodes total ESV (white→red), bottom bar shows land-use fractions
  - Hover: scale up + tooltip with ESV value
- Right panel: **read-only** ESV values per land-use type (Lock icon + "Read-only" label)
  - `?` tooltip on each row shows calibration source
  - When Exp III selected in Model section → crop ESV auto-updates to $332 (+35%) with gold badge
  - Blue info box: "Manual editing with custom retraining will be supported in a future update"
- Legends: color gradient bar + land cover color swatches
- Shared state: `selectedExp` prop from `PageContent.tsx` client wrapper

#### 5. Model Engine (`ModelSection.tsx`) — cream bg
- 3 experiment cards (selectable, with improved text contrast on cream bg):
  - Exp I: Pure eco-value (no_spatial_reward=true)
  - Exp II: Eco-value + spatial rewards (spatial_scale=1.0)
  - Exp III: Spatial + regenerative agriculture (1.35× crops)
- **Expanded config display** (12 parameters from W&B, with hover hints):
  - Reward: reward_scale, spatial_scale
  - Spatial weights: w_tree, w_crop, w_built, w_buf
  - Environment: et_dcs_tolerance, regen_crop, max_steps, n_augment
  - Training: learning_rate, total_timesteps
- W&B link to each experiment's run page
- Phase 1 info box: "Static pre-trained models. Retraining with manual config in future update."
- Load button with loading animation + "Model Loaded" confirmation state
- Shared state: `selectedExp` + `onSelectExp` from `PageContent.tsx`

#### 6. Optimization Results (`ResultsSection.tsx`) — dark bg
- Side-by-side before/after grids with arrow between
  - "After" grid highlights changed cells with green border
- Summary stats: cells changed, total ESV gain, max cell gain, % area modified
- Top-N ESV change rankings table: rank, cell coords, from→to land type, ESV delta

### Architecture Note: Shared Experiment State
`page.tsx` (server component) renders `SideNav` + `PageContent` (client component).
`PageContent.tsx` holds `selectedExp` state and passes it to both `HeatmapSection` and `ModelSection`, enabling the crop ESV auto-update when Exp III is selected.

### File Structure
```
web-app/src/
  app/
    globals.css         ← design tokens (CSS vars), section utilities, heatmap/table styles
    layout.tsx          ← Crimson Pro + Manrope fonts, metadata
    page.tsx            ← server component: SideNav + PageContent
  components/
    PageContent.tsx     ← client wrapper: shared experiment state
    SideNav.tsx         ← fixed right-side dot navigation, IntersectionObserver
    HeroSection.tsx     ← hero with topographic SVG decoration
    ConceptsSection.tsx ← data sources with external links (no land cover legend)
    MapSection.tsx      ← map placeholder + coordinate input
    HeatmapSection.tsx  ← ESV grid + read-only value panel (experiment-aware)
    ModelSection.tsx    ← experiment selector + full W&B config + model loading
    ResultsSection.tsx  ← before/after comparison + rankings table
```

### Status
- [x] Demo layout with mock data — all 6 sections
- [x] Feedback round 1 applied (2026-04-18)
- [x] UI redesign brainstorm — "Scroll-to-Dashboard" approach approved (2026-04-18)
- [ ] Implement dashboard redesign (in progress)
- [ ] Replace map placeholder with Leaflet + click-to-select study area
- [ ] Load actual grid data from API / JSON
- [ ] Connect model inference backend
- [ ] Expand results table to top 50 (currently top 15 in demo)


## Feedback from Ying for Initial Design & Implementation 
0. Remove "CS 8903" above the big title
1. For Research Concepts
- Add link for data source: Sentinel-2: https://climat.esri.ca/datasets/esri::sentinel-2-land-cover-explorer; MODIS: https://planetarycomputer.microsoft.com/explore?c=25.2044%2C-5.0249&z=5.62&v=2; ESVD: https://www.sciencedirect.com/science/article/abs/pii/S0959378014000685 with short proper description and year
- Remove land cover classification part, as it is lack of information
2. For Interactive Map
- Change "Explore Lake Malawi and the surrounding 25km area. Enter coordinates to define a study region, which will be divided into a 50×50 grid for analysis." to "Explore Lake Malawi and the surrounding 25km area. Enter coordinates to define a study region, which represent a 25km*25km area and will be divided into a 50×50 grid for analysis."
3. Economic Value Heatmap
- Make sure when users choose "Exp III", the crop value in this section pop up as 35% increasing value automatically.
- Make sure in product, manual change for users are not allowed and give hints about something like "retraining with manual config values will be supported later and currently is loaded with static models" to guide users.

3. Load Model For Inference
- I don't like current design, it is lack of information. The configuration part should obtain from https://wandb.ai/shakewin/CS8903-odc/runs/2sk0pnp3/overview?nw=nwuseryaoyingshakewin, in configuration, should display settings like reward scale, spatial scale, w_buf, w_built.., et_dcs_tolerance etc with hint of explanation
- The "Pure Eco-Value", "Eco Value + Spatial" words are too dark to see in my system, making ensure it is easier to see and be compatible for both light and dark mode.

## Questions from Ying
- How does the model inference being connected in product? 
- Can we make the heatmap color as same as what is used in backend?
- How grid data is created and loaded when user change the study area by changing the centralized lon/lat? Do we need to create some APIs?

---

## Updates Applied (2026-04-18)

### Changes from Feedback
0. **Hero**: Removed "Georgia Institute of Technology · CS 8903" subtitle line.
1. **Concepts**:
   - Added external links with year for all 3 data sources (Sentinel-2 → Esri Explorer, MODIS → Planetary Computer, ESVD → Costanza 2014 on ScienceDirect).
   - Removed land cover classification pills section entirely.
2. **Map**: Updated description to "...which represent a 25km×25km area and will be divided into a 50×50 grid for analysis."
3. **Heatmap**:
   - ESV inputs are now **read-only** (disabled with cursor-not-allowed styling and Lock icon).
   - When Exp III is selected in the Model section, crop ESV auto-updates from $246 → $332 (+35%) with a gold "+35%" badge and gold info box.
   - Added blue info box: "ESV values are currently loaded from static model configurations. Manual editing with custom retraining will be supported in a future update."
   - Removed the Reset button (no longer needed since values are read-only).
4. **Model**:
   - Expanded config panel from 4 items to **12 W&B parameters**: reward_scale, spatial_scale, w_tree, w_crop, w_built, w_buf, et_dcs_tolerance, regen_crop, learning_rate, total_timesteps, max_steps, n_augment.
   - Each parameter has an info icon with hover tooltip explaining its purpose.
   - Added "View on W&B →" link to each experiment's W&B run page.
   - Actual config values sourced from W&B config.yaml for runs `6f0ta58i` (Exp I), `2sk0pnp3` (Exp II), `pzy2mxod` (Exp III).
   - Experiment cards now render on cream bg with proper text contrast (using `text-text-primary` on `bg-bg-card`), should be readable in both light and dark system modes.
   - Added Phase 1 info box about static models and future retraining support.
5. **Architecture**: Created `PageContent.tsx` client wrapper to share `selectedExp` state between HeatmapSection and ModelSection.

### Answers to Questions

**Q: How does the model inference being connected in product?**
A: Two viable approaches:
1. **API route** (recommended for Phase 1): Export the trained SB3 model to ONNX format, serve it via a Next.js API route (`/api/infer`) or a lightweight FastAPI sidecar. The frontend sends the grid state, the backend runs the model forward pass, and returns the optimized allocation. This keeps the model on the server side and avoids shipping large model weights to the browser.
2. **ONNX in browser** (Phase 2 possibility): Convert the model to ONNX and run inference directly in the browser using `onnxruntime-web`. This eliminates server costs but increases initial page load time (~10MB model download).

For Phase 1, the recommended path is: export to ONNX → FastAPI or Next.js API route → return JSON result.

**Q: Can we make the heatmap color as same as what is used in backend?**
A: Yes. The heatmap currently uses a white→red gradient (`rgb(220,220,220)` → `rgb(255,80,70)`) based on ESV intensity. If the backend (e.g. matplotlib in `eval.py`) uses a specific colormap (like `RdYlGn_r` or a custom one), we can replicate its exact color stops in the frontend. To match precisely: export the colormap's RGB values at key breakpoints and use them in the CSS gradient / JS color interpolation. If you share the exact matplotlib colormap name or hex values, I can update the frontend to match.

**Q: How grid data is created and loaded when user change the study area by changing the centralized lon/lat? Do we need to create some APIs?**
A: Yes, this requires an API. The flow would be:
1. User enters new coordinates in the map section.
2. Frontend calls `POST /api/grid` with `{ lat, lng, year, sample_rate, grid_size, n_rows, n_cols }`.
3. Backend API runs the grid generation logic (similar to what `create_grid()` does in the Python codebase): fetches Sentinel-2 tiles for the new area, computes land cover fractions per cell, and returns the grid as JSON.
4. Frontend receives the grid JSON and renders the heatmap.

This is the most compute-intensive step since it involves downloading and processing satellite imagery. Options to manage latency:
- **Pre-compute**: Cache grids for commonly used coordinates.
- **On-demand**: Run the pipeline per request (~30-60s), show a progress indicator.
- **Hybrid**: Pre-compute a set of grids covering the lake area, serve cached versions when available, and fall back to on-demand for custom coordinates.

---

## UI Redesign: "Scroll-to-Dashboard" (2026-04-18)

### Design Goals
- Transform from a scattered 6-section scroll into an **interactive tool**
- Let users **click on a map** to select study areas instead of typing coordinates
- Keep the map visible while browsing heatmap and results (sticky positioning)
- Merge heatmap, model selection, and results into **one cohesive dashboard**

### Page Structure — Two Zones

**Zone A — Editorial Intro (scrollable, ~1.5 viewports)**
- **Hero**: unchanged (full viewport, scroll cue)
- **Concepts**: compact — 3 data-source cards in a single row, reduced vertical padding (~60% of current)

**Zone B — Interactive Dashboard (sticky map + analysis below)**

```
┌─────────────────────────────────────────────┐
│           MAP (sticky, ~40vh)                │
│   [Leaflet map, click to place study area]  │
│   [25km box overlay + coords display]       │
├─────────────────────────────────────────────┤
│  [ Exp I ]  [ Exp II ]  [ Exp III ]  toolbar│
├────────────────────┬────────────────────────┤
│                    │                        │
│   ESV Heatmap      │   Optimized Result     │
│   (before)         │   (after, highlights)  │
│                    │                        │
├────────────────────┴────────────────────────┤
│  ▶ Summary Stats (cells changed, ESV gain)  │
├─────────────────────────────────────────────┤
│  ▸ Configuration Details  [collapsed]       │
│  ▸ ESV Values & Sources   [collapsed]       │
│  ▸ Change Rankings Table  [collapsed]       │
└─────────────────────────────────────────────┘
```

### Key Interactions
1. **Map click-to-select**: User clicks anywhere on the Leaflet map → 25km×25km bounding box appears centered on click point → coordinates auto-populate in a small overlay on the map
2. **Experiment toolbar**: Compact horizontal tabs (Exp I | II | III). Selecting an experiment instantly updates heatmap ESV values (including Exp III regen crop) and optimized results below
3. **Side-by-side grids**: ESV heatmap (before) on left, optimized result (after) on right. Shared legends below
4. **Summary stats bar**: Cells changed, total ESV gain, max cell gain, % area modified — always visible
5. **Collapsible accordions**: Configuration Details, ESV Values & Sources, Change Rankings Table — collapsed by default, expand on click

### Component Changes
| Component | Action |
|-----------|--------|
| `HeroSection.tsx` | Unchanged |
| `ConceptsSection.tsx` | Reduce padding, keep compact |
| `MapSection.tsx` | Rewrite: Leaflet map, click-to-select, sticky positioning, remove coordinate sidebar |
| `HeatmapSection.tsx` | Extract grid only (no ESV panel), receives experiment state as prop |
| `ModelSection.tsx` | Replace with `ExperimentToolbar.tsx` — compact tab bar |
| `ResultsSection.tsx` | Extract grid + summary stats only, rankings move to accordion |
| `PageContent.tsx` | Becomes `DashboardSection.tsx` — shared state wrapper for all dashboard content |
| `SideNav.tsx` | Simplify to 2 dots: Intro, Dashboard |
| New: `ExperimentToolbar.tsx` | Compact Exp I / II / III tabs with active state |
| New: `AccordionPanel.tsx` | Reusable collapsible panel for config, ESV, rankings |

### Shared State (in DashboardSection)
- `selectedExp`: which experiment is active (drives heatmap ESV + results)
- `studyArea`: `{ lat, lng }` from map click (drives grid data)
- `modelLoaded`: whether inference is ready

### New Dependencies
- `leaflet` + `react-leaflet` for interactive map

### SideNav Update
From 6 dots to 2:
- Intro (covers hero + concepts)
- Dashboard (covers map + all analysis)


## Feedback from Ying for UI Redesign
I like it overall! Just few things:
1. Can the interactive map possibly be able to expand to bigger section but allowing to collap to look like what currently is as well? And shown as expanded section by default. 
2. Can all area except the lake malawi's 25 buffering area be greyed out when user click and not allowing to be chosen and the inference won't process those area?
3. For the Configuration Details and ESV value & resources part, make the text more consistent, such like, "Note: Above are currently loaded from static model configurations. Manual editing with custom retraining will be supported in a future update." And this kind of hint, I prefer to put it as a uppder right question mark icon with hover text in these 2 sections to make page conciser.

### Changes Applied
1. **Map expand/collapse**: Default expanded (60vh), collapse button (bottom-right) shrinks to 30vh. Smooth transition.
2. **Buffer zone restriction**: 25km radius circle shown as dashed green boundary on map. Clicks outside the boundary are ignored. Helper text updated to "Click within the green boundary."
3. **Hint as tooltip**: Removed inline blue info boxes from ConfigPanel and EsvPanel. Added `?` icon (HelpCircle) to the accordion header for "Configuration Details" and "ESV Values & Sources" with hover tooltip showing the note.