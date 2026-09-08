import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, LEVELS } from '../lib/api.js'

export default function Home() {
  const [courses, setCourses] = useState([])
  useEffect(() => { api('/api/courses').then(setCourses).catch(() => {}) }, [])
  const lessons = courses.reduce((a, c) => a + c.lessons, 0)
  return (
    <div className="space-y-14">
      <section className="text-center py-10">
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">
          Master <span className="text-accent">AI/ML Engineering</span>
        </h1>
        <p className="mt-4 text-slate-400 max-w-2xl mx-auto text-lg">
          14 courses · exams after every course · assignments · 50 real-world projects ·
          Copper → Grandmaster levels · an AI tutor that answers every question.
        </p>
        <div className="mt-8 flex justify-center gap-3">
          <Link to="/courses" className="btn">Start learning — it's free</Link>
          <Link to="/login" className="btn-ghost">Create account</Link>
        </div>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        {[['14', 'courses'], [`${lessons}`, 'lessons'], ['50', 'real projects'], ['5', 'levels']].map(([n, l]) => (
          <div key={l} className="card !p-6">
            <div className="text-3xl font-extrabold text-accent">{n}</div>
            <div className="text-sm text-slate-400 mt-1">{l}</div>
          </div>
        ))}
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-6">The level ladder</h2>
        <div className="grid grid-cols-5 gap-3 text-center">
          {LEVELS.map((l, i) => (
            <div key={l.key} className={`rounded-2xl border p-4 ${l.ring.replace('text-', 'text-').split(' ').slice(1).join(' ')}`}>
              <div className="text-2xl">{['🥉', '🥈', '🥇', '💎', '👑'][i]}</div>
              <div className={`font-bold mt-1 ${l.ring.split(' ')[0]}`}>{l.label}</div>
              <div className="text-xs text-slate-400 mt-1">{l.min}+ XP</div>
            </div>
          ))}
        </div>
        <p className="text-center text-xs text-slate-500 mt-3">
          XP comes from passing exams, assignments and completing projects.
        </p>
      </section>

      <section>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">The curriculum</h2>
          <Link to="/courses" className="text-accent text-sm hover:underline">all 14 courses →</Link>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {courses.slice(0, 6).map((c) => (
            <Link key={c.slug} to={`/courses/${c.slug}`} className="card hover:border-accent transition">
              <div className="font-bold">{c.title}</div>
              <p className="text-sm text-slate-400 mt-2 line-clamp-3">{c.description}</p>
              <div className="mt-3 text-xs text-slate-500">{c.lessons} lessons · {c.exams} exam · {c.projects} projects</div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
