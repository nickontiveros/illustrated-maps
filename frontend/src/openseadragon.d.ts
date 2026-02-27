declare namespace OpenSeadragon {
  interface Viewer {
    destroy(): void;
  }
}

declare function OpenSeadragon(options: Record<string, unknown>): OpenSeadragon.Viewer;

declare module 'openseadragon' {
  export = OpenSeadragon;
}
