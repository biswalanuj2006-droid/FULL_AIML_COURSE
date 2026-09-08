import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, LEVELS, levelInfo, nextLevel } from '../lib/api.js'
import { useAuth } from '../auth.jsx'

export default function Dashboard() {
  const { user, ready } = useAuth()
  const [prog, setProg] = useState(null)

  useEffect(() => {
    if (user) api('/api/my/progress').then(setProg).catch(() => {})
  }, [user])

  if (!ready) return null
  if (!user) {
    return (
      <div className="card text-center py-10">
        <p className="text-slate-400">Sign in to track XP, levels and certificates.</p>
        <Link to="/login" className="btn mt-4">Sign in</Link>
      </div>
    )
  }

  const lvl = levelInfo(user.level)
  const nxt = nextLevel(user.xp)
  const base = lvl.min
  const pct = nxt ? Math.min(100, Math.round(100 * (user.xp - base) / (nxt.min - base))) : 100

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-center gap-4">
        <div className={`w-16 h-16 rounded-2xl border-2 flex items-center justify-center text-3xl ${lvl.ring}`}>
          {['🥉', '🥈', '🥇', '💎', '👑'][LEVELS.findIndex((l) => l.key === user.level)]}
        </div>
        <div>
          <h1 className="text-3xl font-extrabold">{user.name}</h1>
          <div className="flex items-center gap-3 mt-1">
            <span className={`badge border ${lvl.ring}`}>{lvl.label} level</span>
            <span className="text-slate-400 text-sm">{user.xp} XP {nxt && <>· {nxt.min - user.xp} XP to {nxt.label}</>}</span>
          </div>
        </div>
      </header>

      <div className="card">
        <div className="flex justify-between text-sm text-slate-400 mb-2">
          <span>{lvl.label}</span>
          {nxt && <span>{nxt.label} at {nxt.min} XP</span>}
        </div>
        <div className="h-3 rounded-full bg-night-700 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-amber-500 via-teal-400 to-fuchsia-500 transition-all" style={{ width: `${pct}%` }} />
        </div>
      </div>

      <section className="grid md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-xl font-bold mb-4">Enrolled courses</h2>
          {prog?.enrollments?.length ? prog.enrollments.map((e) => (
            <div key={e.course_id} className="flex items-center justify-between py-2 border-b border-night-600 last:border-0">
              <span>Course #{e.course_id}</span>
              <span className={`badge ${e.completed ? 'bg-emerald-500/20 text-emerald-300' : 'bg-night-700 text-slate-300'}`}>
                {e.completed ? 'completed 🎉' : `${Math.round(e.progress_pct)}%`}
              </span>
            </div>
          )) : <p className="text-slate-500 text-sm">No enrollments yet — <Link className="text-accent" to="/courses">browse courses</Link>.</p>}
        </div>
        <div className="card">
          <h2 className="text-xl font-bold mb-4">Exam history</h2>
          {prog?.attempts?.length ? prog.attempts.map((a, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b border-night-600 last:border-0">
              <span>Exam #{a.exam_id}</span>
              <span className={`badge ${a.passed ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>{a.score_pct}%</span>
            </div>
          )) : <p className="text-slate-500 text-sm">No exams taken yet.</p>}
        </div>
      </section>
    </div>
  )
}
