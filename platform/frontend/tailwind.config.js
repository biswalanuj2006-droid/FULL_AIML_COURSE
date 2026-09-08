/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        night: { 900: '#0b1220', 800: '#101a2e', 700: '#17243d', 600: '#1f3050' },
        accent: { DEFAULT: '#5eead4', dim: '#2dd4bf' },
      },
    },
  },
  plugins: [],
}
