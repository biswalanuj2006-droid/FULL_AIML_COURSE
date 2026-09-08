import React, { useEffect, useState } from 'react'
import { api, levelInfo } from '../lib/api.js'
import { useAuth } from '../auth.jsx'

export default function Leaderboard() {
  const { user } = useAuth()
  const [rows, setRows] = useState([])
  useEffect(() => { api('/api/leaderboard').then(setRows).catch(() => {}) }, [])
  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-extrabold mb-6">🏆 Leaderboard</h1>
      <div className="card divide-y divide-night-600">
        {rows.map((r, i) => (
          <div key={i} className={`flex items-center gap-4 py-3 ${user && r.name === user.name ? 'text-accent' : ''}`}>
            <span className="w-8 text-center font-bold">{['🥇', '🥈', '🥉'][i] || `#${i + 1}`}</span>
            <span className="flex-1 font-semibold">{r.name}</span>
            <span className={`badge border ${levelInfo(r.level).ring}`}>{levelInfo(r.level).label}</span>
            <span className="w-20 text-right font-mono text-sm">{r.xp} XP</span>
          </div>
        ))}
        {rows.length === 0 && <p className="text-slate-500 py-4">No ranked learners yet.</p>}
      </div>
    </div>
  )
}
