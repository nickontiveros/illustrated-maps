/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'map-primary': '#2563eb',
        'map-secondary': '#4f46e5',
        'map-success': '#16a34a',
        'map-warning': '#ca8a04',
        'map-error': '#dc2626',
      },
    },
  },
  plugins: [],
}
