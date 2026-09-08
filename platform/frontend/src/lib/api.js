// Tiny API + auth helpers for the whole app.
// For static hosting (Vercel/Netlify), set VITE_API_BASE at build time to the
// backend URL (e.g. https://your-app.up.railway.app). Same-origin deploys
// (Railway/Render/Fly/Docker/Heroku) need nothing.
export const API_BASE = import.meta.env.VITE_API_BASE || ''

const TOKEN_KEY = 'aiml_token'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function setToken(t) {
  if (t) localStorage.setItem(TOKEN_KEY, t)
  else localStorage.removeItem(TOKEN_KEY)
}

export async function api(path, { method = 'GET', body } = {}) {
  const headers = { 'Content-Type': 'application/json' }
  const t = getToken()
  if (t) headers.Authorization = `Bearer ${t}`
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const msg = data.detail || `${res.status} ${res.statusText}`
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return data
}

export const LEVELS = [
  { key: 'copper', label: 'Copper', min: 0, ring: 'text-amber-700 border-amber-700/60 bg-amber-900/30' },
  { key: 'silver', label: 'Silver', min: 300, ring: 'text-slate-200 border-slate-300/60 bg-slate-500/20' },
  { key: 'gold', label: 'Gold', min: 800, ring: 'text-yellow-300 border-yellow-400/60 bg-yellow-500/20' },
  { key: 'platinum', label: 'Platinum', min: 1800, ring: 'text-cyan-200 border-cyan-300/60 bg-cyan-500/20' },
  { key: 'grandmaster', label: 'Grandmaster', min: 3500, ring: 'text-fuchsia-300 border-fuchsia-400/60 bg-fuchsia-500/20' },
]

export function levelInfo(key) {
  return LEVELS.find((l) => l.key === key) || LEVELS[0]
}

export function nextLevel(xp) {
  for (const l of LEVELS) if (xp < l.min) return l
  return null
}

export function assetUrl(p) {
  if (!p) return ''
  const base = p.startsWith('http') ? '' : API_BASE
  return p.startsWith('http') || p.startsWith('/assets/') ? `${base}${p}` : `${base}/assets/${p}`
}
