import { useState, useEffect, useCallback } from 'react';
import { useUpdateProject } from '@/hooks/useProjects';
import type {
  StyleSettings,
  BorderSettings,
  NarrativeSettings,
} from '@/types';

type ProjectDetail = import('@/types').ProjectDetail;

interface ProjectSettingsProps {
  project: ProjectDetail;
}

// ── Reusable slider ──────────────────────────────────────────────
function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  suffix = '',
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  suffix?: string;
}) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-500">{label}</span>
        <span className="font-medium">
          {step < 1 ? value.toFixed(2) : value}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

// ── Collapsible section ──────────────────────────────────────────
function Section({
  title,
  enabled,
  onToggle,
  defaultOpen = false,
  children,
}: {
  title: string;
  enabled?: boolean;
  onToggle?: (v: boolean) => void;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen || (enabled ?? true));

  return (
    <section className="border border-slate-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-3 flex items-center justify-between bg-slate-50 hover:bg-slate-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-700">{title}</span>
          {onToggle !== undefined && (
            <label
              className="relative inline-flex items-center cursor-pointer"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => onToggle(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-8 h-4 bg-slate-300 peer-checked:bg-blue-600 rounded-full after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:after:translate-x-full" />
            </label>
          )}
        </div>
        <span className="text-slate-400 text-xs">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className={`p-4 space-y-3 ${enabled === false ? 'opacity-50' : ''}`}>
          {children}
        </div>
      )}
    </section>
  );
}

// ── Main component ───────────────────────────────────────────────
export default function ProjectSettings({ project }: ProjectSettingsProps) {
  const updateProject = useUpdateProject(project.name);

  // Local form state
  const [title, setTitle] = useState(project.title ?? '');
  const [subtitle, setSubtitle] = useState(project.subtitle ?? '');

  // Style settings
  const [palettePreset, setPalettePreset] = useState(project.style.palette_preset ?? '');
  const [paletteStrength, setPaletteStrength] = useState(project.style.palette_enforcement_strength ?? 0);
  const [colorConsistency, setColorConsistency] = useState(project.style.color_consistency_strength ?? 0.5);
  const [terrainExaggeration, setTerrainExaggeration] = useState(project.style.terrain_exaggeration ?? 1.0);

  // Typography
  const [typoEnabled, setTypoEnabled] = useState(project.style.typography?.enabled ?? false);
  const [roadLabels, setRoadLabels] = useState(project.style.typography?.road_labels ?? true);
  const [districtLabels, setDistrictLabels] = useState(project.style.typography?.district_labels ?? true);
  const [waterLabels, setWaterLabels] = useState(project.style.typography?.water_labels ?? true);
  const [parkLabels, setParkLabels] = useState(project.style.typography?.park_labels ?? true);
  const [fontScale, setFontScale] = useState(project.style.typography?.font_scale ?? 1.0);
  const [maxLabels, setMaxLabels] = useState(project.style.typography?.max_labels ?? 200);

  // Road style
  const [roadEnabled, setRoadEnabled] = useState(project.style.road_style?.enabled ?? false);
  const [roadPreset, setRoadPreset] = useState(project.style.road_style?.preset ?? '');
  const [wobble, setWobble] = useState(project.style.road_style?.wobble_amount ?? 1.5);
  const [roadOverlay, setRoadOverlay] = useState(project.style.road_style?.overlay_on_output ?? false);

  // Border
  const [borderEnabled, setBorderEnabled] = useState(project.border?.enabled ?? false);
  const [borderStyle, setBorderStyle] = useState(project.border?.style ?? 'vintage_scroll');
  const [borderMargin, setBorderMargin] = useState(project.border?.margin ?? 200);
  const [showCompass, setShowCompass] = useState(project.border?.show_compass ?? true);
  const [showLegend, setShowLegend] = useState(project.border?.show_legend ?? true);

  // Atmosphere
  const [atmoEnabled, setAtmoEnabled] = useState(project.style.atmosphere?.enabled ?? false);
  const [hazeStrength, setHazeStrength] = useState(project.style.atmosphere?.haze_strength ?? 0.3);
  const [contrastReduction, setContrastReduction] = useState(project.style.atmosphere?.contrast_reduction ?? 0.2);

  // Narrative
  const [narrativeAutoDiscover, setNarrativeAutoDiscover] = useState(project.narrative?.auto_discover ?? false);
  const [narrativeMaxLandmarks, setNarrativeMaxLandmarks] = useState(project.narrative?.max_landmarks ?? 50);
  const [narrativeMinScore, setNarrativeMinScore] = useState(project.narrative?.min_importance_score ?? 0.3);

  // Reset form when project changes
  useEffect(() => {
    setTitle(project.title ?? '');
    setSubtitle(project.subtitle ?? '');
    setPalettePreset(project.style.palette_preset ?? '');
    setPaletteStrength(project.style.palette_enforcement_strength ?? 0);
    setColorConsistency(project.style.color_consistency_strength ?? 0.5);
    setTerrainExaggeration(project.style.terrain_exaggeration ?? 1.0);
    setTypoEnabled(project.style.typography?.enabled ?? false);
    setRoadLabels(project.style.typography?.road_labels ?? true);
    setDistrictLabels(project.style.typography?.district_labels ?? true);
    setWaterLabels(project.style.typography?.water_labels ?? true);
    setParkLabels(project.style.typography?.park_labels ?? true);
    setFontScale(project.style.typography?.font_scale ?? 1.0);
    setMaxLabels(project.style.typography?.max_labels ?? 200);
    setRoadEnabled(project.style.road_style?.enabled ?? false);
    setRoadPreset(project.style.road_style?.preset ?? '');
    setWobble(project.style.road_style?.wobble_amount ?? 1.5);
    setRoadOverlay(project.style.road_style?.overlay_on_output ?? false);
    setBorderEnabled(project.border?.enabled ?? false);
    setBorderStyle(project.border?.style ?? 'vintage_scroll');
    setBorderMargin(project.border?.margin ?? 200);
    setShowCompass(project.border?.show_compass ?? true);
    setShowLegend(project.border?.show_legend ?? true);
    setAtmoEnabled(project.style.atmosphere?.enabled ?? false);
    setHazeStrength(project.style.atmosphere?.haze_strength ?? 0.3);
    setContrastReduction(project.style.atmosphere?.contrast_reduction ?? 0.2);
    setNarrativeAutoDiscover(project.narrative?.auto_discover ?? false);
    setNarrativeMaxLandmarks(project.narrative?.max_landmarks ?? 50);
    setNarrativeMinScore(project.narrative?.min_importance_score ?? 0.3);
  }, [project]);

  // Build the update payload from form state
  const buildPayload = useCallback(() => {
    const style: StyleSettings = {
      ...project.style,
      palette_preset: palettePreset || null,
      palette_enforcement_strength: paletteStrength,
      color_consistency_strength: colorConsistency,
      terrain_exaggeration: terrainExaggeration,
      typography: {
        enabled: typoEnabled,
        road_labels: roadLabels,
        district_labels: districtLabels,
        water_labels: waterLabels,
        park_labels: parkLabels,
        title_text: null,
        subtitle_text: null,
        font_scale: fontScale,
        halo_width: project.style.typography?.halo_width ?? 2,
        max_labels: maxLabels,
        min_road_length_px: project.style.typography?.min_road_length_px ?? 100,
      },
      road_style: {
        enabled: roadEnabled,
        motorway_exaggeration: project.style.road_style?.motorway_exaggeration ?? 20,
        primary_exaggeration: project.style.road_style?.primary_exaggeration ?? 15,
        secondary_exaggeration: project.style.road_style?.secondary_exaggeration ?? 10,
        residential_exaggeration: project.style.road_style?.residential_exaggeration ?? 5,
        motorway_color: project.style.road_style?.motorway_color ?? null,
        primary_color: project.style.road_style?.primary_color ?? null,
        secondary_color: project.style.road_style?.secondary_color ?? null,
        residential_color: project.style.road_style?.residential_color ?? null,
        outline_color: project.style.road_style?.outline_color ?? null,
        wobble_amount: wobble,
        wobble_frequency: project.style.road_style?.wobble_frequency ?? 0.02,
        overlay_on_output: roadOverlay,
        overlay_opacity: project.style.road_style?.overlay_opacity ?? 0.3,
        reference_opacity: project.style.road_style?.reference_opacity ?? 0.7,
        preset: roadPreset || null,
      },
      atmosphere: {
        enabled: atmoEnabled,
        haze_color: project.style.atmosphere?.haze_color ?? '#C8D8E8',
        haze_strength: hazeStrength,
        contrast_reduction: contrastReduction,
        saturation_reduction: project.style.atmosphere?.saturation_reduction ?? 0.15,
        gradient_curve: project.style.atmosphere?.gradient_curve ?? 1.5,
      },
    };

    const border: BorderSettings = {
      enabled: borderEnabled,
      style: borderStyle,
      margin: borderMargin,
      show_compass: showCompass,
      show_legend: showLegend,
      show_scale_bar: false,
      border_color: project.border?.border_color ?? null,
      background_color: project.border?.background_color ?? null,
      ornament_opacity: project.border?.ornament_opacity ?? 0.8,
    };

    const narrative: NarrativeSettings = {
      auto_discover: narrativeAutoDiscover,
      max_landmarks: narrativeMaxLandmarks,
      show_activities: project.narrative?.show_activities ?? false,
      max_activity_markers: project.narrative?.max_activity_markers ?? 100,
      min_importance_score: narrativeMinScore,
    };

    return {
      style,
      title: title || undefined,
      subtitle: subtitle || undefined,
      border,
      narrative,
    };
  }, [
    project, title, subtitle, palettePreset, paletteStrength, colorConsistency,
    terrainExaggeration, typoEnabled, roadLabels, districtLabels, waterLabels,
    parkLabels, fontScale, maxLabels, roadEnabled, roadPreset, wobble, roadOverlay,
    borderEnabled, borderStyle, borderMargin, showCompass, showLegend,
    atmoEnabled, hazeStrength, contrastReduction,
    narrativeAutoDiscover, narrativeMaxLandmarks, narrativeMinScore,
  ]);

  // Dirty detection
  const isDirty = (() => {
    const t = project;
    if ((title || '') !== (t.title ?? '')) return true;
    if ((subtitle || '') !== (t.subtitle ?? '')) return true;
    if ((palettePreset || '') !== (t.style.palette_preset ?? '')) return true;
    if (paletteStrength !== (t.style.palette_enforcement_strength ?? 0)) return true;
    if (colorConsistency !== (t.style.color_consistency_strength ?? 0.5)) return true;
    if (terrainExaggeration !== (t.style.terrain_exaggeration ?? 1.0)) return true;
    if (typoEnabled !== (t.style.typography?.enabled ?? false)) return true;
    if (roadLabels !== (t.style.typography?.road_labels ?? true)) return true;
    if (districtLabels !== (t.style.typography?.district_labels ?? true)) return true;
    if (waterLabels !== (t.style.typography?.water_labels ?? true)) return true;
    if (parkLabels !== (t.style.typography?.park_labels ?? true)) return true;
    if (fontScale !== (t.style.typography?.font_scale ?? 1.0)) return true;
    if (maxLabels !== (t.style.typography?.max_labels ?? 200)) return true;
    if (roadEnabled !== (t.style.road_style?.enabled ?? false)) return true;
    if ((roadPreset || '') !== (t.style.road_style?.preset ?? '')) return true;
    if (wobble !== (t.style.road_style?.wobble_amount ?? 1.5)) return true;
    if (roadOverlay !== (t.style.road_style?.overlay_on_output ?? false)) return true;
    if (borderEnabled !== (t.border?.enabled ?? false)) return true;
    if (borderStyle !== (t.border?.style ?? 'vintage_scroll')) return true;
    if (borderMargin !== (t.border?.margin ?? 200)) return true;
    if (showCompass !== (t.border?.show_compass ?? true)) return true;
    if (showLegend !== (t.border?.show_legend ?? true)) return true;
    if (atmoEnabled !== (t.style.atmosphere?.enabled ?? false)) return true;
    if (hazeStrength !== (t.style.atmosphere?.haze_strength ?? 0.3)) return true;
    if (contrastReduction !== (t.style.atmosphere?.contrast_reduction ?? 0.2)) return true;
    if (narrativeAutoDiscover !== (t.narrative?.auto_discover ?? false)) return true;
    if (narrativeMaxLandmarks !== (t.narrative?.max_landmarks ?? 50)) return true;
    if (narrativeMinScore !== (t.narrative?.min_importance_score ?? 0.3)) return true;
    return false;
  })();

  const handleSave = async () => {
    const payload = buildPayload();
    await updateProject.mutateAsync(payload);
  };

  return (
    <div className="p-4 space-y-4">
      {/* Title Section */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold text-slate-700">Title</h3>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Map title"
          className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
        <input
          type="text"
          value={subtitle}
          onChange={(e) => setSubtitle(e.target.value)}
          placeholder="Subtitle"
          className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </section>

      {/* Style Section */}
      <Section title="Style & Color" defaultOpen>
        <div>
          <label className="block text-sm text-slate-500 mb-1">Palette Preset</label>
          <select
            value={palettePreset}
            onChange={(e) => setPalettePreset(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg"
          >
            <option value="">None</option>
            <option value="vintage_tourist">Vintage Tourist</option>
            <option value="modern_pop">Modern Pop</option>
            <option value="ink_wash">Ink Wash</option>
          </select>
        </div>
        <Slider label="Palette Enforcement" value={paletteStrength} onChange={setPaletteStrength} min={0} max={1} step={0.05} />
        <Slider label="Color Consistency" value={colorConsistency} onChange={setColorConsistency} min={0} max={1} step={0.05} />
        <Slider label="Terrain Exaggeration" value={terrainExaggeration} onChange={setTerrainExaggeration} min={1} max={5} step={0.1} suffix="x" />
      </Section>

      {/* Typography Section */}
      <Section title="Typography" enabled={typoEnabled} onToggle={setTypoEnabled}>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={roadLabels} onChange={(e) => setRoadLabels(e.target.checked)} />
            Road labels
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={districtLabels} onChange={(e) => setDistrictLabels(e.target.checked)} />
            District labels
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={waterLabels} onChange={(e) => setWaterLabels(e.target.checked)} />
            Water labels
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={parkLabels} onChange={(e) => setParkLabels(e.target.checked)} />
            Park labels
          </label>
        </div>
        <Slider label="Font Scale" value={fontScale} onChange={setFontScale} min={0.5} max={3} step={0.1} suffix="x" />
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-500">Max Labels</span>
            <span className="font-medium">{maxLabels}</span>
          </div>
          <input
            type="number"
            min={0}
            max={500}
            value={maxLabels}
            onChange={(e) => setMaxLabels(parseInt(e.target.value) || 0)}
            className="w-full px-3 py-1 text-sm border border-slate-300 rounded"
          />
        </div>
      </Section>

      {/* Road Style Section */}
      <Section title="Road Style" enabled={roadEnabled} onToggle={setRoadEnabled}>
        <div>
          <label className="block text-sm text-slate-500 mb-1">Preset</label>
          <select
            value={roadPreset}
            onChange={(e) => setRoadPreset(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg"
          >
            <option value="">Custom</option>
            <option value="vintage_tourist">Vintage Tourist</option>
            <option value="modern_clean">Modern Clean</option>
            <option value="ink_sketch">Ink Sketch</option>
          </select>
        </div>
        <Slider label="Wobble" value={wobble} onChange={setWobble} min={0} max={5} step={0.1} suffix="px" />
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={roadOverlay} onChange={(e) => setRoadOverlay(e.target.checked)} />
          Overlay on output
        </label>
      </Section>

      {/* Border Section */}
      <Section title="Border" enabled={borderEnabled} onToggle={setBorderEnabled}>
        <div>
          <label className="block text-sm text-slate-500 mb-1">Style</label>
          <select
            value={borderStyle}
            onChange={(e) => setBorderStyle(e.target.value as BorderSettings['style'])}
            className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg"
          >
            <option value="vintage_scroll">Vintage Scroll</option>
            <option value="art_deco">Art Deco</option>
            <option value="modern_minimal">Modern Minimal</option>
            <option value="ornate_victorian">Ornate Victorian</option>
          </select>
        </div>
        <Slider label="Margin" value={borderMargin} onChange={(v) => setBorderMargin(Math.round(v))} min={50} max={500} step={10} suffix="px" />
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={showCompass} onChange={(e) => setShowCompass(e.target.checked)} />
          Show compass rose
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={showLegend} onChange={(e) => setShowLegend(e.target.checked)} />
          Show legend
        </label>
      </Section>

      {/* Atmosphere Section */}
      <Section title="Atmosphere" enabled={atmoEnabled} onToggle={setAtmoEnabled}>
        <Slider label="Haze Strength" value={hazeStrength} onChange={setHazeStrength} min={0} max={1} step={0.05} />
        <Slider label="Contrast Reduction" value={contrastReduction} onChange={setContrastReduction} min={0} max={0.5} step={0.05} />
      </Section>

      {/* Narrative/Discovery Section */}
      <Section title="Landmark Discovery" defaultOpen={false}>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={narrativeAutoDiscover} onChange={(e) => setNarrativeAutoDiscover(e.target.checked)} />
          Auto-discover landmarks
        </label>
        <Slider label="Min Importance Score" value={narrativeMinScore} onChange={setNarrativeMinScore} min={0} max={1} step={0.05} />
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-500">Max Landmarks</span>
            <span className="font-medium">{narrativeMaxLandmarks}</span>
          </div>
          <input
            type="number"
            min={0}
            max={200}
            value={narrativeMaxLandmarks}
            onChange={(e) => setNarrativeMaxLandmarks(parseInt(e.target.value) || 0)}
            className="w-full px-3 py-1 text-sm border border-slate-300 rounded"
          />
        </div>
      </Section>

      {/* Info sections (read-only) */}
      <Section title="Region" defaultOpen={false}>
        <div className="bg-slate-50 rounded-lg p-3 text-sm">
          <div className="grid grid-cols-2 gap-2">
            <div>North: {project.region.north.toFixed(4)}</div>
            <div>South: {project.region.south.toFixed(4)}</div>
            <div>East: {project.region.east.toFixed(4)}</div>
            <div>West: {project.region.west.toFixed(4)}</div>
          </div>
        </div>
      </Section>

      <Section title="Output" defaultOpen={false}>
        <div className="bg-slate-50 rounded-lg p-3 text-sm space-y-1">
          <div>Size: {project.output.width}x{project.output.height}px</div>
          <div>DPI: {project.output.dpi}</div>
        </div>
      </Section>

      <Section title="Tiles" defaultOpen={false}>
        <div className="bg-slate-50 rounded-lg p-3 text-sm space-y-1">
          <div>Size: {project.tiles.size}px</div>
          <div>Overlap: {project.tiles.overlap}px</div>
          <div>Grid: {project.grid_cols}x{project.grid_rows}</div>
          <div>Total: {project.tile_count} tiles</div>
        </div>
      </Section>

      <Section title="Detail Level" defaultOpen={false}>
        <div className="bg-slate-50 rounded-lg p-3 text-sm">
          {project.detail_level}
        </div>
      </Section>

      {/* Save button */}
      {isDirty && (
        <div className="sticky bottom-0 bg-white pt-2 pb-4 border-t border-slate-200">
          <button
            onClick={handleSave}
            disabled={updateProject.isPending}
            className="w-full px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {updateProject.isPending ? 'Saving...' : 'Save Changes'}
          </button>
          {updateProject.isError && (
            <div className="mt-2 text-sm text-red-600">
              Failed to save: {(updateProject.error as Error).message}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
