# MapGen Frontend

React-based visual frontend for the Illustrated Map Generator.

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

## Requirements

The frontend requires the MapGen API backend running at http://localhost:8000

Start the backend with:

```bash
cd /home/nmo/maps
pip install -e ".[api]"
uvicorn mapgen.api.main:app --reload
```

## Features

- **Project Management**: Create, view, and delete map projects
- **Tile Grid Visualization**: Interactive grid showing generation status
- **Geographic Map View**: Leaflet-based map with project bounds
- **Deep Zoom Viewer**: OpenSeadragon for viewing large assembled images
- **Generation Progress**: Real-time WebSocket updates during tile generation
- **Seam Repair**: View and repair tile boundary artifacts
- **Landmark Management**: Add, position, and illustrate landmarks

## Tech Stack

- React 18
- TypeScript
- Vite
- TailwindCSS
- React Query (TanStack Query)
- Zustand (state management)
- Leaflet (geographic maps)
- OpenSeadragon (deep zoom images)
